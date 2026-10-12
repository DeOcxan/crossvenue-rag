from crossvenue_rag.reporting.compute.distinct import shared_value


def test_the_shared_value_is_the_one_distinct_value_and_none_otherwise():
    assert shared_value([3, 3, 3]) == 3
    assert shared_value([3, 4]) is None
    assert shared_value([]) is None
