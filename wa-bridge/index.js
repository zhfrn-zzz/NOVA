import makeWASocket, {
    useMultiFileAuthState,
    DisconnectReason,
    fetchLatestBaileysVersion,
} from "@whiskeysockets/baileys";
import express from "express";
import pino from "pino";
import { Boom } from "@hapi/boom";
import qrcode from "qrcode-terminal";

const PORT = process.env.WA_BRIDGE_PORT || 3001;

// Allowed JIDs — supports both @lid and @s.whatsapp.net formats
// Set in .env as comma-separated: "628xxx:0@lid,628yyy@s.whatsapp.net"
// Bare phone numbers (e.g. "6282246965391") also work and match @s.whatsapp.net
const RAW_ALLOWED = (process.env.WA_ALLOWED_JIDS || "")
    .split(",")
    .filter(Boolean);

// Pre-expand each entry into all match keys (bare numbers ↔ @s.whatsapp.net)
const ALLOWED_KEYS = new Set();
for (const entry of RAW_ALLOWED) {
    ALLOWED_KEYS.add(entry);
    if (entry.endsWith("@s.whatsapp.net")) {
        ALLOWED_KEYS.add(entry.split("@")[0]);
    } else if (!entry.includes("@")) {
        ALLOWED_KEYS.add(`${entry}@s.whatsapp.net`);
    }
}

function isAllowedJid(jid) {
    if (ALLOWED_KEYS.size === 0) return true; // no filter = accept all
    // Check the JID itself and (for @s.whatsapp.net) the bare phone number
    if (ALLOWED_KEYS.has(jid)) return true;
    if (jid.endsWith("@s.whatsapp.net") && ALLOWED_KEYS.has(jid.split("@")[0])) {
        return true;
    }
    return false;
}

const logger = pino({ level: "warn" });
const app = express();
app.use(express.json());

let sock = null;
let pendingCallback = null;

// --- Baileys Connection ---

async function connectWA() {
    const { state, saveCreds } = await useMultiFileAuthState("./auth");
    const { version } = await fetchLatestBaileysVersion();

    sock = makeWASocket({
        version,
        auth: state,
        logger,
        printQRInTerminal: false,
    });

    sock.ev.on("creds.update", saveCreds);

    sock.ev.on("connection.update", (update) => {
        const { connection, lastDisconnect, qr } = update;

        if (qr) {
            console.log("\n=== Scan QR Code with WhatsApp ===");
            qrcode.generate(qr, { small: true });
        }

        if (connection === "close") {
            const reason = new Boom(lastDisconnect?.error)?.output?.statusCode;
            if (reason === DisconnectReason.loggedOut) {
                console.log("Logged out. Delete auth/ folder and restart.");
            } else {
                console.log(
                    `Disconnected (reason: ${reason}), reconnecting...`
                );
                setTimeout(connectWA, 3000);
            }
        }

        if (connection === "open") {
            console.log("WhatsApp connected!");
        }
    });

    // --- Incoming Messages ---
    sock.ev.on("messages.upsert", async ({ messages }) => {
        for (const msg of messages) {
            if (msg.key.fromMe) continue;

            const jid = msg.key.remoteJid;
            if (!jid) continue;

            // Authorization check — smart match (supports @lid, @s.whatsapp.net, and bare numbers)
            if (!isAllowedJid(jid)) {
                console.log(
                    `Unauthorized message from ${jid}, ignoring`
                );
                continue;
            }

            const text =
                msg.message?.conversation ||
                msg.message?.extendedTextMessage?.text ||
                "";

            if (!text.trim()) continue;

            console.log(`Message from ${jid}: ${text}`);

            // Forward to NOVA via callback
            if (pendingCallback) {
                try {
                    const resp = await fetch(pendingCallback, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            sender: jid,
                            text,
                            jid,
                        }),
                    });
                    const data = await resp.json();

                    if (data.reply) {
                        await sock.sendMessage(msg.key.remoteJid, {
                            text: data.reply,
                        });
                    }
                } catch (err) {
                    console.error("Error forwarding to NOVA:", err.message);
                    await sock.sendMessage(msg.key.remoteJid, {
                        text: "NOVA sedang tidak bisa memproses. Coba lagi nanti.",
                    });
                }
            }
        }
    });
}

// --- HTTP API for NOVA ---

app.post("/register-callback", (req, res) => {
    pendingCallback = req.body.url;
    console.log(`Callback registered: ${pendingCallback}`);
    res.json({ ok: true });
});

app.post("/send", async (req, res) => {
    const { to, text } = req.body;
    if (!sock) return res.status(503).json({ error: "WhatsApp not connected" });

    try {
        // Accept raw JID or phone number
        const jid = to.includes("@") ? to : `${to}@s.whatsapp.net`;
        await sock.sendMessage(jid, { text });
        res.json({ ok: true });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get("/health", (req, res) => {
    res.json({
        connected: sock?.user ? true : false,
        user: sock?.user?.id || null,
    });
});

// --- Start ---
app.listen(PORT, () => {
    console.log(`WA Bridge API running on http://localhost:${PORT}`);
    connectWA();
});
