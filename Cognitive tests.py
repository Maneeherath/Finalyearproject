# ============================================================
#  PAGE 5: COGNITIVE TEST BATTERY
#  Tests: Reaction Time, Memory Sequence, Attention Count
#  Results feed into the ML prediction model
# ============================================================

import streamlit as st
import time, random

def show_cognitive_tests():
    st.title("🧠 Cognitive Assessment Battery")
    st.markdown("Complete all 3 tests below. Your results will be used to **improve prediction accuracy** by measuring your actual cognitive performance rather than estimating it.")
    st.info("💡 Take these tests honestly — they directly replace the estimated scores in the model.")
    st.divider()

    # ── Initialize session state ─────────────────────────────
    for key, default in [
        ('reaction_times', []),
        ('reaction_phase', 'instructions'),  # instructions → waiting → ready → done
        ('reaction_start', 0),
        ('reaction_complete', False),
        ('memory_sequence', []),
        ('memory_phase', 'show'),
        ('memory_input', ''),
        ('memory_score', None),
        ('memory_attempts', 0),
        ('attention_score', None),
        ('attention_phase', 'instructions'),
        ('attention_sequence', []),
        ('all_tests_done', False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ════════════════════════════════════════════════════════
    #  TEST 1: REACTION TIME
    # ════════════════════════════════════════════════════════
    with st.expander("⚡ Test 1: Reaction Time", expanded=True):
        st.markdown("**Instructions:** Click **'I'm Ready'**, then click **'CLICK NOW!'** as fast as you can when the green button appears. You will do this 3 times.")

        if not st.session_state['reaction_complete']:
            attempt = len(st.session_state['reaction_times']) + 1

            if st.session_state['reaction_phase'] == 'instructions':
                if st.button("▶️ Start Reaction Test", key="rt_start"):
                    st.session_state['reaction_phase'] = 'waiting'
                    st.rerun()

            elif st.session_state['reaction_phase'] == 'waiting':
                st.warning(f"⏳ Attempt {attempt}/3 — Get ready... wait for the green button!")
                # Random delay using attempt number as seed for reproducibility
                delay = random.uniform(1.5, 4.0)
                time.sleep(delay)
                st.session_state['reaction_phase'] = 'ready'
                st.session_state['reaction_start'] = time.time()
                st.rerun()

            elif st.session_state['reaction_phase'] == 'ready':
                st.success("🟢 **CLICK NOW!**")
                if st.button("🟢 CLICK NOW!", key=f"rt_click_{attempt}", type="primary"):
                    rt = round((time.time() - st.session_state['reaction_start']) * 1000)
                    st.session_state['reaction_times'].append(rt)
                    if len(st.session_state['reaction_times']) >= 3:
                        st.session_state['reaction_phase'] = 'done'
                        st.session_state['reaction_complete'] = True
                    else:
                        st.session_state['reaction_phase'] = 'waiting'
                    st.rerun()

        if st.session_state['reaction_complete']:
            times = st.session_state['reaction_times']
            avg_rt = round(sum(times) / len(times))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Attempt 1", f"{times[0]} ms")
            c2.metric("Attempt 2", f"{times[1]} ms")
            c3.metric("Attempt 3", f"{times[2]} ms")
            c4.metric("⭐ Average", f"{avg_rt} ms")

            if avg_rt < 250:
                st.success("🟢 Excellent reaction time!")
            elif avg_rt < 350:
                st.info("🟡 Average reaction time.")
            else:
                st.warning("🔴 Slow reaction time — may indicate fatigue.")

    # ════════════════════════════════════════════════════════
    #  TEST 2: MEMORY SEQUENCE
    # ════════════════════════════════════════════════════════
    with st.expander("🔢 Test 2: Memory Sequence", expanded=True):
        st.markdown("**Instructions:** A sequence of numbers will be shown for **3 seconds**. Memorize them, then type them back in order.")

        if st.session_state['memory_score'] is None:
            # Generate sequence once
            if not st.session_state['memory_sequence']:
                st.session_state['memory_sequence'] = [random.randint(1,9) for _ in range(7)]
                st.session_state['memory_phase'] = 'show'

            seq = st.session_state['memory_sequence']
            seq_str = ' - '.join(map(str, seq))

            if st.session_state['memory_phase'] == 'show':
                st.markdown(f"### 👀 Memorize this sequence:")
                st.markdown(f"## `{seq_str}`")
                st.caption("You have 5 seconds to memorize it.")
                if st.button("✅ I've memorized it — hide the sequence", key="mem_hide"):
                    st.session_state['memory_phase'] = 'recall'
                    st.rerun()

            elif st.session_state['memory_phase'] == 'recall':
                st.markdown("### ✍️ Type the sequence (digits separated by spaces):")
                user_input = st.text_input("Your answer:", placeholder="e.g. 3 7 1 9 4 2 8",
                                           key="mem_input_field")
                if st.button("Submit Answer", key="mem_submit"):
                    try:
                        user_nums = list(map(int, user_input.strip().split()))
                        correct   = sum(a==b for a,b in zip(user_nums, seq))
                        score     = round((correct / len(seq)) * 100)
                        st.session_state['memory_score'] = score
                        st.rerun()
                    except:
                        st.error("Please enter numbers separated by spaces (e.g. 3 7 1 9 4 2 8)")

        if st.session_state['memory_score'] is not None:
            score = st.session_state['memory_score']
            seq   = st.session_state['memory_sequence']
            st.markdown(f"**Correct sequence was:** `{' - '.join(map(str, seq))}`")
            st.metric("Memory Score", f"{score}%")
            if score >= 80:
                st.success("🟢 Excellent memory performance!")
            elif score >= 50:
                st.info("🟡 Average memory performance.")
            else:
                st.warning("🔴 Below average — consider getting more sleep.")

    # ════════════════════════════════════════════════════════
    #  TEST 3: ATTENTION TEST
    # ════════════════════════════════════════════════════════
    with st.expander("👁️ Test 3: Attention Test", expanded=True):
        st.markdown("**Instructions:** A grid of emojis will appear. Count how many 🎯 targets you can find and enter the number.")

        if st.session_state['attention_score'] is None:
            if not st.session_state['attention_sequence']:
                targets = random.randint(5, 12)
                distractors = 25 - targets
                items = ['🎯'] * targets + ['⭐'] * distractors
                random.shuffle(items)
                st.session_state['attention_sequence'] = items
                st.session_state['attention_target_count'] = targets

            items   = st.session_state['attention_sequence']
            correct = st.session_state.get('attention_target_count', 0)

            st.markdown("### Find all the 🎯 targets:")
            rows = [items[i:i+5] for i in range(0, 25, 5)]
            for row in rows:
                st.markdown("  ".join(row))

            count_input = st.number_input("How many 🎯 did you count?",
                                          min_value=0, max_value=25, value=0,
                                          key="att_input")
            if st.button("Submit Count", key="att_submit"):
                diff  = abs(count_input - correct)
                score = max(0, round((1 - diff / correct) * 100)) if correct > 0 else 0
                st.session_state['attention_score'] = score
                st.session_state['attention_correct'] = correct
                st.rerun()

        if st.session_state['attention_score'] is not None:
            score   = st.session_state['attention_score']
            correct = st.session_state.get('attention_correct', 0)
            st.markdown(f"**Correct count was:** {correct} targets")
            st.metric("Attention Score", f"{score}%")
            if score >= 80:
                st.success("🟢 Excellent attention performance!")
            elif score >= 50:
                st.info("🟡 Average attention performance.")
            else:
                st.warning("🔴 Below average — may indicate attention difficulties.")

    # ════════════════════════════════════════════════════════
    #  COMBINED RESULTS
    # ════════════════════════════════════════════════════════
    st.divider()
    rt_done  = st.session_state['reaction_complete']
    mem_done = st.session_state['memory_score'] is not None
    att_done = st.session_state['attention_score'] is not None

    st.subheader("📋 Test Progress")
    c1, c2, c3 = st.columns(3)
    c1.metric("⚡ Reaction Test", "✅ Done" if rt_done  else "⏳ Pending")
    c2.metric("🔢 Memory Test",   "✅ Done" if mem_done else "⏳ Pending")
    c3.metric("👁️ Attention Test","✅ Done" if att_done else "⏳ Pending")

    if rt_done and mem_done and att_done:
        st.success("🎉 All tests complete! Your results have been saved.")

        times  = st.session_state['reaction_times']
        avg_rt = round(sum(times)/len(times))
        mem_s  = st.session_state['memory_score']
        att_s  = st.session_state['attention_score']

        # Save to session state for use in prediction
        st.session_state['cognitive_results'] = {
            'reaction_time_ms': avg_rt,
            'memory_score_pct': mem_s,
            'attention_score_pct': att_s
        }

        st.divider()
        st.subheader("🧠 Your Cognitive Profile")
        col1, col2, col3 = st.columns(3)
        col1.metric("Reaction Time",  f"{avg_rt} ms",
                    "Fast ✅" if avg_rt < 300 else "Slow ⚠️")
        col2.metric("Memory Score",   f"{mem_s}%",
                    "Good ✅" if mem_s >= 70 else "Low ⚠️")
        col3.metric("Attention Score",f"{att_s}%",
                    "Good ✅" if att_s >= 70 else "Low ⚠️")

        st.info("➡️ Go to **🏠 Home & Prediction** — your measured scores will now be used instead of estimated ones!")

        if st.button("🔄 Reset All Tests", key="reset_tests"):
            for key in ['reaction_times','reaction_phase','reaction_start',
                        'reaction_complete','memory_sequence','memory_phase',
                        'memory_score','memory_attempts','attention_score',
                        'attention_phase','attention_sequence','cognitive_results']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()