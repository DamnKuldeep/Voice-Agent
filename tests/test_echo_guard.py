from voiceagent.pipeline.echo_guard import EchoVerdict, echo_verdict

AGENT = "The capital of France is Paris, a city on the Seine."


def test_the_agents_own_words_are_echo():
    assert echo_verdict("capital of France is", AGENT) is EchoVerdict.ECHO


def test_new_words_are_not_echo():
    assert echo_verdict("no wait tell me about spain", AGENT) is EchoVerdict.CLEAR
    assert echo_verdict("wait Paris", AGENT) is EchoVerdict.CLEAR   # half new is a real interruption


def test_one_matching_word_waits_for_more_unless_the_candidate_is_over():
    assert echo_verdict("Paris", AGENT) is EchoVerdict.UNDECIDED
    assert echo_verdict("Paris", AGENT, final=True) is EchoVerdict.ECHO


def test_nothing_to_compare_is_clear():
    assert echo_verdict("stop", "") is EchoVerdict.CLEAR
    assert echo_verdict("", AGENT) is EchoVerdict.CLEAR
