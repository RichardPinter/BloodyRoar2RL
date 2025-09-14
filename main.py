#!/usr/bin/env python3
"""
Main orchestrator for the RL game agent.
Starts and coordinates all components.
"""
import time
import csv
import threading
import signal
import sys
from src.config import *
from src.logging_utils import log_state
from src.shared_state import SharedState
from src.screen_capture import ScreenCapture
from src.agent import GameAgent
from src.trainer import Trainer

class RLGameAgent:
    """Main orchestrator class that manages all components"""
    
    def __init__(self):
        log_state("="*60)
        if ENABLE_LOGGING:
            log_state(f"Starting RL agent - Log file: {LOG_FILENAME}")
        else:
            log_state("Starting RL agent - File logging DISABLED")
        
        self.log_startup_info()
        
        # Initialize shared state
        self.shared_state = SharedState()
        
        # Initialize components
        self.screen_capture = ScreenCapture(
            self.shared_state.frame_queue,
            self.shared_state.stop_event
        )
        
        self.agent = GameAgent(self.shared_state)
        self.trainer = Trainer(self.shared_state)
        # self.validation_gui = RoundValidationGUI()
        
        # Thread handles
        self.threads = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        log_state("="*60)
    
    def log_startup_info(self):
        """Log configuration information at startup"""
        log_state(f"Device: {DEVICE}")
        log_state(f"Region: {REGION}")
        log_state(f"Health bar locations - P1: {X1_P1}-{X2_P1}, P2: {X1_P2}-{X2_P2}")
        log_state(f"Learning rate: {LEARNING_RATE}")
        log_state(f"Learning starts after: {LEARNING_STARTS} transitions (warm-up)")
        log_state(f"Soft target update: tau={TAU}")
        log_state(f"Round detection: State-independent with confirmation")
        log_state(f"Round indicators: {len(ROUND_INDICATORS)} positions")
        
        if LOAD_CHECKPOINT:
            log_state(f"Checkpoint: {LOAD_CHECKPOINT}")
        
        if TEST_MODE:
            log_state("MODE: TEST (training disabled)")
        else:
            log_state("MODE: TRAINING")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        log_state("\nShutdown signal received...")
        self.shutdown()
        sys.exit(0)
    
    def start_threads(self):
        """Start all worker threads"""
        # Start screen capture
        self.screen_capture.start()
        
        # Start agent thread
        agent_thread = threading.Thread(
            target=self.agent.run,
            name="Agent",
            daemon=True
        )
        agent_thread.start()
        self.threads.append(agent_thread)
        
        # Start trainer thread
        trainer_thread = threading.Thread(
            target=self.trainer.run,
            name="Trainer",
            daemon=True
        )
        trainer_thread.start()
        self.threads.append(trainer_thread)
        
        # Start validation GUI thread
        # gui_thread = threading.Thread(
        #     target=self.validation_gui.run_gui_thread,
        #     name="ValidationGUI",
        #     daemon=True
        # )
        # gui_thread.start()
        # self.threads.append(gui_thread)
        
        log_state("All threads started successfully")
    
    def run(self):
        """Main run loop"""
        self.start_threads()
        
        try:
            # Main loop - just keep alive and monitor
            while True:
                time.sleep(1)
                
                # Optional: Add periodic status logging here
                if self.shared_state.global_step % 3600 == 0 and self.shared_state.global_step > 0:
                    self.log_status()
                
        except KeyboardInterrupt:
            log_state("\nKeyboard interrupt received")
            self.shutdown()
    
    def log_status(self):
        """Log current system status"""
        log_state(f"Status: Step {self.shared_state.global_step}, "
                 f"Episode {self.shared_state.episode_number}, "
                 f"Match {self.shared_state.match_number}, "
                 f"Buffer size {self.shared_state.replay_buffer.len}")
    
    def shutdown(self):
        """Clean shutdown of all components"""
        log_state("Shutting down...")
        
        # Signal all threads to stop
        self.shared_state.shutdown()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                log_state(f"Warning: Thread {thread.name} did not stop cleanly")
        
        # Wait for screen capture
        self.screen_capture.join()
        
        # Save final results
        self.save_results()
        
        log_state("Shutdown complete")
    
    def save_results(self):
        """Save final results to CSV"""
        # Save health readings
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "p1_pct", "p2_pct"])
            writer.writerows(self.shared_state.results)
        
        log_state(f"Saved {len(self.shared_state.results)} health readings")
        
        if ENABLE_LOGGING:
            log_state(f"Log file saved to: {LOG_FILENAME}")

def main():
    """Entry point"""
    agent = RLGameAgent()
    agent.run()

if __name__ == "__main__":
    main()