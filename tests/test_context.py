"""Tests for ConversationContext sliding window."""

from nova.memory.context import ConversationContext


class TestConversationContext:
    def test_empty_context(self):
        ctx = ConversationContext()
        assert ctx.get_context() == []
        assert ctx.turn_count == 0

    def test_add_single_exchange(self):
        ctx = ConversationContext()
        ctx.add_exchange("hello", "hi there")
        context = ctx.get_context()
        assert len(context) == 2
        assert context[0] == {"role": "user", "content": "hello"}
        assert context[1] == {"role": "assistant", "content": "hi there"}
        assert ctx.turn_count == 1

    def test_add_multiple_exchanges(self):
        ctx = ConversationContext()
        ctx.add_exchange("msg1", "resp1")
        ctx.add_exchange("msg2", "resp2")
        ctx.add_exchange("msg3", "resp3")
        assert ctx.turn_count == 3
        assert len(ctx.get_context()) == 6

    def test_sliding_window_trims_oldest(self):
        ctx = ConversationContext(max_turns=2)
        ctx.add_exchange("msg1", "resp1")
        ctx.add_exchange("msg2", "resp2")
        ctx.add_exchange("msg3", "resp3")  # should push out msg1/resp1

        context = ctx.get_context()
        assert ctx.turn_count == 2
        assert len(context) == 4
        # Oldest exchange should be gone
        assert context[0] == {"role": "user", "content": "msg2"}
        assert context[1] == {"role": "assistant", "content": "resp2"}
        assert context[2] == {"role": "user", "content": "msg3"}
        assert context[3] == {"role": "assistant", "content": "resp3"}

    def test_sliding_window_with_max_turns_1(self):
        ctx = ConversationContext(max_turns=1)
        ctx.add_exchange("a", "b")
        ctx.add_exchange("c", "d")
        context = ctx.get_context()
        assert len(context) == 2
        assert context[0]["content"] == "c"
        assert context[1]["content"] == "d"

    def test_clear_resets_context(self):
        ctx = ConversationContext()
        ctx.add_exchange("hello", "world")
        ctx.clear()
        assert ctx.get_context() == []
        assert ctx.turn_count == 0

    def test_get_context_returns_copy(self):
        ctx = ConversationContext()
        ctx.add_exchange("hello", "world")
        context = ctx.get_context()
        context.clear()  # modify returned list
        assert ctx.turn_count == 1  # original unchanged
