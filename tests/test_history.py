from voiceagent.pipeline.history import INTERRUPTED_MARK, History


def fill(history: History, pairs: int) -> None:
    for i in range(pairs):
        history.add_user(f"q{i}")
        history.add_assistant(f"a{i}")


def test_keeps_only_the_last_turns():
    history = History(turns=2)
    fill(history, 5)
    assert [m["content"] for m in history.messages()] == ["q3", "a3", "q4", "a4"]


def test_window_never_starts_with_an_assistant_message():
    history = History(turns=1)
    history.add_user("q0")
    history.add_assistant("a0")
    history.add_user("q1")
    assert [m["role"] for m in history.messages()] == ["user"]


def test_interrupted_reply_is_cut_to_the_spoken_words_and_flagged():
    history = History(turns=3)
    history.add_user("tell me about paris")
    history.add_assistant("Paris is big. It has a river. It is old.")
    history.mark_interrupted("Paris is big.")
    assistant = history.messages()[-1]
    assert assistant["content"] == f"Paris is big. {INTERRUPTED_MARK}"


def test_interrupted_before_any_audio_drops_the_reply():
    history = History(turns=3)
    history.add_user("hello")
    history.add_assistant("Hi there, how can I help?")
    history.mark_interrupted("  ")
    assert [m["role"] for m in history.messages()] == ["user"]


def test_drop_last_user_and_clear():
    history = History(turns=3)
    fill(history, 1)
    history.add_user("oops")
    history.drop_last_user()
    assert len(history.messages()) == 2
    history.drop_last_user()
    assert len(history.messages()) == 2  # last item is an assistant message: untouched
    history.clear()
    assert history.messages() == []
