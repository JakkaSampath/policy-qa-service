from app.guards import AnswerCache, RateLimiter, SlotLimiter, normalize


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# ---- rate limiter ----

def test_rate_limiter_allows_limit_then_blocks_with_retry_time():
    clock = Clock()
    rl = RateLimiter(limit=3, window_s=60, clock=clock)
    assert [rl.check("a") for _ in range(3)] == [None, None, None]
    clock.now += 10
    assert rl.check("a") == 50  # oldest request leaves the window in 50 s


def test_rate_limiter_window_slides():
    clock = Clock()
    rl = RateLimiter(limit=2, window_s=60, clock=clock)
    rl.check("a"); rl.check("a")
    clock.now += 60
    assert rl.check("a") is None


def test_rate_limiter_is_per_client():
    rl = RateLimiter(limit=1, window_s=60, clock=Clock())
    assert rl.check("a") is None
    assert rl.check("a") is not None
    assert rl.check("b") is None


def test_rejected_requests_do_not_extend_the_block():
    clock = Clock()
    rl = RateLimiter(limit=1, window_s=60, clock=clock)
    rl.check("a")
    for _ in range(5):
        clock.now += 10
        rl.check("a")          # rejected, must not be recorded
    clock.now += 10            # 60 s after the only accepted request
    assert rl.check("a") is None


def test_rate_limiter_forgets_idle_clients():
    clock = Clock()
    rl = RateLimiter(limit=5, window_s=60, clock=clock, max_clients=2)
    rl.check("a"); rl.check("b")
    clock.now += 61
    rl.check("c")              # table full -> idle a, b are dropped
    assert set(rl._hits) == {"c"}


# ---- answer slots ----

def test_slot_limiter_turns_away_when_full():
    slots = SlotLimiter(2)
    assert slots.try_acquire() and slots.try_acquire()
    assert not slots.try_acquire()
    slots.release()
    assert slots.try_acquire()


# ---- cache ----

def test_normalize_ignores_case_spaces_and_trailing_punctuation():
    assert normalize("  How MUCH   per mile?? ") == normalize("how much per mile") == "how much per mile"


def test_cache_hit_on_normalized_question():
    cache = AnswerCache(size=4)
    cache.put("Sick leave days?", "A")
    assert cache.get("  sick   LEAVE days ") == "A"
    assert cache.get("annual leave days") is None


def test_cache_evicts_least_recently_used():
    cache = AnswerCache(size=2)
    cache.put("q1", "A"); cache.put("q2", "B")
    cache.get("q1")            # q1 is now most recent
    cache.put("q3", "C")       # evicts q2
    assert (cache.get("q1"), cache.get("q2"), cache.get("q3")) == ("A", None, "C")


def test_cache_size_zero_disables_it():
    cache = AnswerCache(size=0)
    cache.put("q", "A")
    assert cache.get("q") is None
