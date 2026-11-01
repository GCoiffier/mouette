(function () {
  const scrollingClass = "md-sidebar__scrollwrap--scrolling";
  const canScrollClass = "md-sidebar__scrollwrap--can-scroll";
  const atTopClass = "md-sidebar__scrollwrap--at-top";
  const atBottomClass = "md-sidebar__scrollwrap--at-bottom";

  // Two short windows protect the primary sidebar from getting forced back to
  // top by Material's instant-nav while still letting the user scroll freely.
  //
  //   NAV_SETTLE_MS: after a navigation event, allow any programmatic scroll
  //   adjustments through. Material applies a few post-nav scroll tweaks that
  //   we want to land before locking position preservation.
  //
  //   USER_SCROLL_MS: after a real user input (wheel, touch, pointer,
  //   keyboard), trust subsequent scroll events as user-driven. Without this
  //   window momentum scrolling on macOS gets misclassified as programmatic.
  //
  // Both windows are anchored to real events (document$ emission, user
  // input). The values are conservative upper bounds rather than tuned
  // magic numbers.
  const NAV_SETTLE_MS = 400;
  const USER_SCROLL_MS = 700;

  const timers = new WeakMap();
  const locks = new WeakMap();

  function getLock(scrollwrap) {
    let lock = locks.get(scrollwrap);
    if (!lock) {
      lock = {
        allowProgrammaticUntil: 0,
        lastTop: scrollwrap.scrollTop,
        restoring: false,
        userScrollUntil: 0,
      };
      locks.set(scrollwrap, lock);
    }

    return lock;
  }

  function isPrimary(scrollwrap) {
    return !!scrollwrap.closest(".md-sidebar--primary");
  }

  function markUserScroll(scrollwrap) {
    getLock(scrollwrap).userScrollUntil =
      window.performance.now() + USER_SCROLL_MS;
  }

  function markNavigationSettle(scrollwrap) {
    getLock(scrollwrap).allowProgrammaticUntil =
      window.performance.now() + NAV_SETTLE_MS;
  }

  function updateScrollState(scrollwrap) {
    const maxScroll = Math.max(0, scrollwrap.scrollHeight - scrollwrap.clientHeight);
    const scrollTop = scrollwrap.scrollTop;
    const canScroll = maxScroll > 1;

    scrollwrap.classList.toggle(canScrollClass, canScroll);
    scrollwrap.classList.toggle(atTopClass, !canScroll || scrollTop <= 1);
    scrollwrap.classList.toggle(atBottomClass, !canScroll || scrollTop >= maxScroll - 1);
  }

  function setScrolling(scrollwrap) {
    updateScrollState(scrollwrap);
    scrollwrap.classList.add(scrollingClass);

    const previousTimer = timers.get(scrollwrap);
    if (previousTimer) {
      window.clearTimeout(previousTimer);
    }

    timers.set(
      scrollwrap,
      window.setTimeout(() => {
        scrollwrap.classList.remove(scrollingClass);
      }, 500)
    );
  }

  function handleScroll(scrollwrap) {
    const lock = getLock(scrollwrap);
    const time = window.performance.now();
    const primary = isPrimary(scrollwrap);
    const userDriven = time <= lock.userScrollUntil;
    const allowProgrammatic = time <= lock.allowProgrammaticUntil;

    if (
      primary &&
      !lock.restoring &&
      !userDriven &&
      !allowProgrammatic &&
      Math.abs(scrollwrap.scrollTop - lock.lastTop) > 1
    ) {
      lock.restoring = true;
      scrollwrap.scrollTop = lock.lastTop;
      updateScrollState(scrollwrap);
      lock.restoring = false;
      return;
    }

    setScrolling(scrollwrap);

    if (!primary || userDriven || allowProgrammatic) {
      lock.lastTop = scrollwrap.scrollTop;
    }
  }

  function bindScrollwrap(scrollwrap) {
    if (scrollwrap.dataset.alloyScrollBound) {
      updateScrollState(scrollwrap);
      return;
    }

    scrollwrap.dataset.alloyScrollBound = "true";
    updateScrollState(scrollwrap);
    window.requestAnimationFrame(() => updateScrollState(scrollwrap));
    if (isPrimary(scrollwrap)) {
      const mark = () => markUserScroll(scrollwrap);
      scrollwrap.addEventListener("wheel", mark, { passive: true });
      scrollwrap.addEventListener("touchstart", mark, { passive: true });
      scrollwrap.addEventListener("pointerdown", mark, { passive: true });
      scrollwrap.addEventListener("keydown", mark);
      window.requestAnimationFrame(() => {
        getLock(scrollwrap).lastTop = scrollwrap.scrollTop;
      });
    }

    scrollwrap.addEventListener("scroll", () => handleScroll(scrollwrap), {
      passive: true,
    });

    markNavigationSettle(scrollwrap);
  }

  function bindAll() {
    document
      .querySelectorAll(".md-sidebar__scrollwrap")
      .forEach(bindScrollwrap);
  }

  function markAllNavigationSettle() {
    document
      .querySelectorAll(".md-sidebar__scrollwrap")
      .forEach(markNavigationSettle);
  }

  bindAll();
  window.addEventListener("resize", bindAll, { passive: true });
  if (window.document$ && typeof window.document$.subscribe === "function") {
    // Material's instant-nav emits on document$ when navigation completes.
    // Re-bind any new scrollwraps and open the settle window so post-nav
    // adjustments land before position preservation kicks back in.
    window.document$.subscribe(() => {
      bindAll();
      markAllNavigationSettle();
    });
  }
})();
