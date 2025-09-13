#!/usr/bin/env python3
"""
Game agent (consumer thread) that processes frames and makes decisions.
This is the main game-playing logic.
"""
import time
import cv2
import numpy as np
import torch
import random
from collections import deque
from config import *
from logging_utils import *
from game_vision import *
from round_tracking import RoundState, MatchTracker
from torch.utils.tensorboard import SummaryWriter

class GameAgent:
    """Main game-playing agent that consumes frames and makes decisions"""
    
    def __init__(self, shared_state):
        self.shared = shared_state
        self.writer = SummaryWriter(log_dir=f"{LOG_DIR}/tensorboard")
        
        # Initialize tracking components
        self.round_state = RoundState()
        self.match_tracker = MatchTracker(start_match_number=self.shared.match_number)
        
        # State machine
        self.state = "waiting_for_match"
        self.state_before_fallback = None
        
        # IMPORTANT: Local frame stack, not shared!
        self.frame_stack = deque(maxlen=FRAME_STACK)
        
        # Timing trackers
        self.alive_since = None
        self.death_since = None
        self.both_at_zero_since = None
        self.health_lost_since = None
        self.last_health_log_time = time.time()
        
        # Fallback mode
        self.fallback_action_count = 0
        self.post_match_action_count = 0
        self.post_match_entry_logged = False
        
        # RL state tracking
        self.prev_state = None
        self.prev_action = None
        self.prev_extra_feats = None
        self.prev_pct1 = 100.0
        self.prev_pct2 = 100.0
        
        # Action control
        self.hold_counter = 0
        self.current_action = None
        
        # Episode tracking
        self.round_reward = 0.0
        self.round_start_time = time.time()
        self.round_steps = 0
        self.action_counts = np.zeros(NUM_ACTIONS)
        self.reward_history = deque(maxlen=100)
        
        # Episode counting
        self.episode_count = 1
        
        # CV2 windows
        cv2.namedWindow("Q-Values", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Rewards", cv2.WINDOW_AUTOSIZE)
        self._position_windows()
        
        log_state("GameAgent initialized")
    
    def _position_windows(self):
        """Position debug windows below capture area"""
        x, y, w, h = REGION
        win_x = x
        win_y = y + h + 100
        cv2.moveWindow("Q-Values", win_x, win_y)
        cv2.moveWindow("Rewards", win_x + 300, win_y)
    
    def write_action(self, text: str):
        """Write action to file for game input"""
        try:
            with open(ACTIONS_FILE, "w") as f:
                f.write(text)
        except Exception as e:
            log_debug(f"ERROR writing action: {e}")
    
    def handle_menu_navigation(self, state_name="menu", action_count=0):
        """Handle menu navigation by alternating kick/start"""
        current_time = time.time()
        time_slot = int(current_time * 2)  # Changes every 0.5 seconds
        
        if time_slot % 2 == 0:
            action_to_write = "kick\n"
        else:
            action_to_write = "start\n"
        
        action_count += 1
        
        if action_count % 50 == 0:
            log_debug(f"{state_name} navigation #{action_count}: {action_to_write.strip()}")
        
        self.write_action(action_to_write)
        return action_to_write, action_count
    
    def on_round_end(self, winner):
        """Handle round end logic"""
        log_round(f"[Episode] Round #{self.episode_count} total_reward={self.round_reward:.2f}")
        self.episode_count += 1
        episode_num = self.shared.increment_episode_number()
        
        # Log episode metrics
        round_duration = time.time() - self.round_start_time
        self.writer.add_scalar("episode/reward", self.round_reward, episode_num)
        self.writer.add_scalar("episode/length_steps", self.round_steps, episode_num)
        self.writer.add_scalar("episode/length_seconds", round_duration, episode_num)
        self.writer.add_scalar("episode/win", 1.0 if winner == "p1" else 0.0, episode_num)
        
        # Log action distribution
        if self.action_counts.sum() > 0:
            action_probs = self.action_counts / self.action_counts.sum()
            for i, action_name in enumerate(ACTIONS):
                self.writer.add_scalar(f"actions/episode_distribution/{action_name}",
                                      action_probs[i], episode_num)
        
        # Update reward history and display
        self.reward_history.append(self.round_reward)
        # self.draw_reward_graph()
        
        # Final transition with terminal reward
        if self.prev_state is not None and self.prev_action is not None:
            final_reward = FINAL_REWARD if winner == "p1" else -FINAL_REWARD
            zero_next_state = np.zeros_like(self.prev_state, dtype=np.float32)
            zero_next_extras = np.zeros_like(self.prev_extra_feats, dtype=np.float32)
            self.shared.replay_buffer.add(
                self.prev_state, self.prev_extra_feats, self.prev_action,
                final_reward, zero_next_state, zero_next_extras, True
            )
        
        # Signal round end for learner
        self.shared.signal_round_end()
        
        # Reset round tracking
        self.round_reward = 0.0
    
    def draw_reward_graph(self):
        """Draw reward history graph"""
        h, w = 150, 300
        graph = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(self.reward_history) > 1:
            mn = min(self.reward_history)
            mx = max(self.reward_history)
            span = mx - mn if mx != mn else 1.0
            pts = []
            for i, r in enumerate(self.reward_history):
                x = int(i * (w-1) / (len(self.reward_history)-1))
                y = h - 1 - int((r - mn) * (h-1) / span)
                pts.append((x, y))
            cv2.polylines(graph, [np.array(pts, np.int32)], False, (0,255,0), 2)
            
            # Draw min/max labels
            cv2.putText(graph, f"{mx:.1f}", (5,15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
            cv2.putText(graph, f"{mn:.1f}", (5,h-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)
        
        cv2.imshow("Rewards", graph)
        cv2.waitKey(1)
    
    def display_q_values(self, q_vals):
        """Display Q-values for debugging"""
        disp = np.zeros((20 * NUM_ACTIONS, 200, 3), dtype=np.uint8)
        for i, (act, val) in enumerate(zip(ACTIONS, q_vals)):
            y = 15 + i * 20
            cv2.putText(disp, f"{act[:6]:6s}: {val:6.2f}", (5, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
        cv2.imshow("Q-Values", disp)
        cv2.waitKey(1)
    
    def select_action(self, state_tensor, extras_tensor, legal_mask):
        """Select action using epsilon-greedy policy with legal action masking"""
        legal_idx = np.flatnonzero(legal_mask)
        if legal_idx.size == 0:
            legal_mask[:] = True
            legal_idx = np.arange(NUM_ACTIONS)
        
        eps = 0.01 if TEST_MODE else max(0.01, 0.30 - 0.29 * (self.shared.replay_buffer.len / 20000))
        
        with torch.no_grad():
            q = self.shared.policy_net(state_tensor, extras_tensor)
            q = q.masked_fill(torch.tensor(~legal_mask, device=q.device).unsqueeze(0),
                            float("-inf"))
            # Display Q-values
            # self.display_q_values(q.squeeze(0).cpu().numpy())
        
        if random.random() < eps:
            return int(np.random.choice(legal_idx))
        else:
            return int(q.argmax(1).item())
    
    def run(self):
        """Main agent loop"""
        log_state("Agent thread started", self.state)
      
        
        # Stub for validation GUI if not available
        try:
            from validation_gui import RoundValidationGUI
            self.validation_gui = RoundValidationGUI()
        except:
            class ValidationStub:
                def request_validation(self, **kwargs):
                    pass
            self.validation_gui = ValidationStub()
        
        while not self.shared.stop_event.is_set():
            # Get frame with timeout
            frame = None
            ts = None
            try:
                frame, ts = self.shared.frame_queue.get(timeout=0.1)
            except:
                # No frame - handle menu navigation in certain states
                if self.state == "post_match_waiting":
                    _, self.post_match_action_count = self.handle_menu_navigation(
                        "post_match_no_frame", self.post_match_action_count)
                elif self.state == "health_detection_lost":
                    _, self.fallback_action_count = self.handle_menu_navigation(
                        "fallback_no_frame", self.fallback_action_count)
                continue
            
            # Process frame
            self.process_frame(frame, ts)
            # Mark task done
            try:
                self.shared.frame_queue.task_done()
            except:
                pass
        
        # Cleanup
        cv2.destroyAllWindows()
        self.writer.close()
        log_state("Agent thread stopped")
    
    def process_frame(self, frame, ts):
        """Process a single frame - main game logic"""
        self.shared.increment_global_step()
        
        # Detect health
        pct1, pct2 = detect_health(frame)
        
        # Log health periodically
        current_time = time.time()
        if current_time - self.last_health_log_time >= 1.0:
            self.last_health_log_time = current_time
        
        # Check for health restoration (round start detection)
        health_restoration_detected = self.check_health_restoration(pct1, pct2)
        
        # Handle health detection fallback
        self.handle_health_fallback(pct1, pct2)
        
        if self.state == "health_detection_lost":
            _, self.fallback_action_count = self.handle_menu_navigation(
                "health_fallback", self.fallback_action_count)
            return
        
        # Detect round indicators
        round_indicators, indicator_states = detect_round_indicators(frame)
        detected_p1_rounds = sum([round_indicators['p1_round1'], round_indicators['p1_round2']])
        detected_p2_rounds = sum([round_indicators['p2_round1'], round_indicators['p2_round2']])
        
        # Update round state
        round_result = self.round_state.update_first_to_zero(
            pct1, pct2,
            validation_callback=self.validation_gui.request_validation
        )
        
        # Handle fallback if needed
        if round_result and round_result[0] == "fallback_needed":
            round_result = self.handle_round_fallback(
                round_indicators, detected_p1_rounds, detected_p2_rounds, pct1, pct2)
        
        # Process round end if detected
        if round_result and round_result[0] == "round_won":
            self.handle_round_completion(round_result)
        
        # Handle state-specific logic
        if self.state == "waiting_for_match":
            self.handle_waiting_for_match(pct1, pct2, health_restoration_detected)
        elif self.state == "waiting_for_round":
            self.handle_waiting_for_round(pct1, pct2)
        elif self.state == "post_match_waiting":
            self.handle_post_match(detected_p1_rounds, detected_p2_rounds, 
                                  indicator_states, pct1, pct2)
        elif self.state == "active":
            self.handle_active_state(frame, pct1, pct2)
        # Save results
        if frame is not None:
            self.shared.results.append((time.time(), pct1, pct2))
            if len(self.shared.screenshots) < MAX_FRAMES:
                self.shared.screenshots.append(frame.copy())
    
    def check_health_restoration(self, pct1, pct2):
        """Check for health restoration indicating round start"""
        # Check if both players are at 0% health
        if pct1 <= DEATH_THRESHOLD and pct2 <= DEATH_THRESHOLD:
            if self.both_at_zero_since is None:
                self.both_at_zero_since = time.time()
                log_state(f"⚠️ Both players at 0% health - tracking...")
        else:
            if self.both_at_zero_since is not None:
                time_at_zero = time.time() - self.both_at_zero_since
                
                # If both are now alive AND were dead for required duration
                if (pct1 >= ALIVE_THRESHOLD and pct2 >= ALIVE_THRESHOLD and
                    time_at_zero >= ZERO_HEALTH_DURATION):
                    log_round(f"🎯 ROUND START DETECTED via health restoration!")
                    self.both_at_zero_since = None
                    return True
                
                self.both_at_zero_since = None
        
        return False
    
    def handle_health_fallback(self, pct1, pct2):
        """Handle health detection fallback logic"""
        if pct1 == 0.0 and pct2 == 0.0:
            if self.health_lost_since is None:
                self.health_lost_since = time.time()
                log_state("⚠️ Health bars lost - starting timer...")
            else:
                time_lost = time.time() - self.health_lost_since
                if time_lost >= HEALTH_LOST_TIMEOUT and self.state != "health_detection_lost":
                    self.state_before_fallback = self.state
                    log_state(f"🚨 HEALTH DETECTION LOST - Entering fallback mode!")
                    self.state = "health_detection_lost"
                    self.fallback_action_count = 0
                    self.alive_since = None
                    self.death_since = None
                    self.current_action = None
                    self.hold_counter = 0
        else:
            # Health bars detected - check if we need to exit fallback
            if self.state == "health_detection_lost":
                if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
                    log_state(f"✅ Health bars restored! Exiting fallback mode")
                    self.recover_from_fallback(pct1, pct2)
    
    def recover_from_fallback(self, pct1, pct2):
        """Recover from health detection fallback"""
        self.round_state.clear_all_candidates()
        
        # Determine appropriate state
        if self.state_before_fallback == "post_match_waiting":
            self.state = "post_match_waiting"
        else:
            confirmed_p1, confirmed_p2 = self.round_state.get_current_state()
            if confirmed_p1 == 0 and confirmed_p2 == 0:
                self.state = "waiting_for_match"
            else:
                self.state = "waiting_for_round"
        
        self.health_lost_since = None
        self.fallback_action_count = 0
        self.state_before_fallback = None
    
    def handle_round_fallback(self, round_indicators, detected_p1_rounds, 
                            detected_p2_rounds, pct1, pct2):
        """Use fallback indicator-based detection"""
        log_round("🔄 Using fallback indicator-based detection...")
        old_result = self.round_state.update(detected_p1_rounds, detected_p2_rounds)
        
        if old_result and old_result[0] == "round_won":
            _, winner, p1_rounds, p2_rounds = old_result
            log_round(f"✅ Fallback successful: {winner.upper()} wins via indicators!")
            
            self.validation_gui.request_validation(
                system_prediction=f"{winner.upper()}_FALLBACK",
                p1_health=pct1,
                p2_health=pct2,
                first_to_zero_flag="FALLBACK",
                both_at_zero=True
            )
            
            return old_result
        
        log_round("❌ Fallback failed - no clear winner")
        return None
    
    def handle_round_completion(self, round_result):
        """Handle round completion logic"""
        _, winner, p1_rounds, p2_rounds = round_result
        
        self.on_round_end(winner)
        
        # Check for match end
        match_result = self.match_tracker.check_match_end(p1_rounds, p2_rounds)
        
        if match_result and match_result[0] == "match_over":
            self.shared.signal_match_end()
            log_state(f"🎯 Match over! Entering post-match navigation...")
            self.state = "post_match_waiting"
            self.round_state.clear_all_candidates()
            self.alive_since = None
            self.death_since = None
        else:
            log_state(f"Round ended, waiting for next round...")
            self.state = "waiting_for_round"
            self.alive_since = None
            self.death_since = None
            self.post_match_entry_logged = False
            self.post_match_action_count = 0
    
    def handle_waiting_for_match(self, pct1, pct2, health_restoration_detected):
        """Handle waiting for first round of match"""
        if health_restoration_detected:
            log_state("Health restoration confirms round start")
            self.start_round()
            return
        
        if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
            if not self.round_state.has_round_recently_ended(timeout=1.0):
                self.alive_since = self.alive_since or time.time()
                if time.time() - self.alive_since >= 0.5:
                    log_round("🚀 FIRST ROUND OF MATCH STARTED!")
                    self.start_round()
            else:
                self.alive_since = None
        else:
            self.alive_since = None
    
    def handle_waiting_for_round(self, pct1, pct2):
        """Handle waiting between rounds"""
        if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT:
            self.alive_since = self.alive_since or time.time()
            if time.time() - self.alive_since >= 0.3:
                log_round("🚀 NEXT ROUND STARTED!")
                self.start_round()
        else:
            self.alive_since = None
    
    def handle_post_match(self, detected_p1_rounds, detected_p2_rounds, 
                         indicator_states, pct1, pct2):
        """Handle post-match menu navigation"""
        if not self.post_match_entry_logged:
            self.post_match_entry_logged = True
            log_state("ENTERED POST_MATCH_WAITING STATE")
            self.post_match_action_count = 0
        
        # Check for match reset
        indicators_clear = (detected_p1_rounds == 0 and detected_p2_rounds == 0)
        all_blue = all(state == 'blue' for state in indicator_states.values())
        
        if pct1 >= HEALTH_LIMIT and pct2 >= HEALTH_LIMIT and (indicators_clear or all_blue):
            self.alive_since = self.alive_since or time.time()
            if time.time() - self.alive_since >= 0.5:
                log_state(f"🆕 NEW MATCH DETECTED! Starting Match #{self.match_tracker.match_number}")
                self.round_state.reset()
                self.post_match_entry_logged = False
                self.post_match_action_count = 0
                self.state = "waiting_for_match"
                self.alive_since = None
        else:
            self.alive_since = None
        
        # Navigate menus
        _, self.post_match_action_count = self.handle_menu_navigation(
            "post_match", self.post_match_action_count)
    
    def start_round(self):
        """Initialize a new round"""
        self.state = "active"
        self.frame_stack.clear()
        self.prev_state = None
        self.prev_action = None
        self.prev_extra_feats = None
        self.prev_pct1 = 100.0
        self.prev_pct2 = 100.0
        self.round_start_time = time.time()
        self.round_steps = 0
        self.round_reward = 0.0
        self.action_counts.fill(0)
        self.alive_since = None
        self.post_match_entry_logged = False
        self.post_match_action_count = 0
        
        # Pick a random initial action (not "start")
        # This will be used until we make our first real decision
        self.current_action = random.randint(0, NUM_ACTIONS - 1)  # Random action to start
        self.hold_counter = 0  # Will make a real decision after 4 frames
        
        log_state(f"Round started - initial action: {ACTIONS[self.current_action]}")

    def handle_active_state(self, frame, pct1, pct2):
        self.round_steps += 1

        # Preprocess and add frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, CNN_SIZE, interpolation=cv2.INTER_NEAREST)
        self.frame_stack.append(img.astype(np.float32) / 255.0)

        # --- NEW: if stack isn't full yet, pad with the latest frame so we can decide ---
        if len(self.frame_stack) < FRAME_STACK:
            last = self.frame_stack[-1]
            while len(self.frame_stack) < FRAME_STACK:
                self.frame_stack.append(last)
        # Time to (re)decide?
        if self.hold_counter <= 0 and len(self.frame_stack) >= FRAME_STACK:
            current_state = np.stack(self.frame_stack, 0)
            current_extras = compute_extra_features(frame).astype(np.float32)
            # reward for previous action
            if self.prev_state is not None and self.prev_action is not None:
                our_damage = self.prev_pct1 - pct1
                opp_damage = self.prev_pct2 - pct2
                reward = np.clip(opp_damage - our_damage, -REWARD_CLIP, REWARD_CLIP)
                self.round_reward += reward
                self.shared.replay_buffer.add(
                    self.prev_state, self.prev_extra_feats, self.prev_action,
                    reward, current_state, current_extras, False
                )
            # choose action
            state_tensor = torch.from_numpy(current_state).unsqueeze(0).to(DEVICE)
            extras_tensor = torch.from_numpy(current_extras).unsqueeze(0).to(DEVICE)
            ts = classify_transform_state(frame)
            legal_mask = legal_mask_from_ts(ts)

            self.current_action = self.select_action(state_tensor, extras_tensor, legal_mask)
            self.action_counts[self.current_action] += 1
            self.hold_counter = HOLD_FRAMES

            # track for next reward
            self.prev_state = current_state
            self.prev_extra_feats = current_extras
            self.prev_action = self.current_action
            self.prev_pct1 = pct1
            self.prev_pct2 = pct2

            log_debug(f"Decision made: {ACTIONS[self.current_action]} at step {self.round_steps}")
        # always send current action
        self.write_action(ACTIONS[self.current_action] + "\n")

        if self.hold_counter > 0:
            self.hold_counter -= 1
