#!/usr/bin/env python3
"""
GUI for manual round validation.
Allows human confirmation of round winners for training accuracy.
"""
import os
import csv
import tkinter as tk
from tkinter import ttk
from queue import Queue
from datetime import datetime
from config import ROUND_VALIDATION_LOG
from logging_utils import log_state, log_round

class RoundValidationGUI:
    def __init__(self):
        self.validation_queue = Queue()
        self.result_queue = Queue()
        self.validation_log = ROUND_VALIDATION_LOG
        
        # Initialize CSV file
        if not os.path.exists(self.validation_log):
            with open(self.validation_log, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'system_prediction', 'human_confirmation',
                    'p1_health', 'p2_health', 'first_to_zero_flag',
                    'both_at_zero', 'system_correct'
                ])
        
        log_state("🎯 Round validation GUI initialized")
    
    def request_validation(self, system_prediction, p1_health, p2_health,
                          first_to_zero_flag, both_at_zero):
        """Queue a validation request (called from main thread)"""
        validation_data = {
            'timestamp': datetime.now().isoformat(),
            'system_prediction': system_prediction,
            'p1_health': p1_health,
            'p2_health': p2_health,
            'first_to_zero_flag': first_to_zero_flag,
            'both_at_zero': both_at_zero
        }
        self.validation_queue.put(validation_data)
        log_round(f"🤔 Manual validation requested: System says {system_prediction} wins")
    
    def show_validation_dialog(self, validation_data):
        """Show the validation dialog (runs in GUI thread)"""
        root = tk.Tk()
        root.title("Round Validation")
        root.geometry("400x300")
        root.attributes('-topmost', True)  # Always on top
        root.focus_force()
        
        # Center the window
        root.eval('tk::PlaceWindow . center')
        
        selected_winner = tk.StringVar(value="")
        result = {'confirmed': False, 'winner': None}
        
        # Title
        title_label = tk.Label(root, text="🎯 ROUND ENDED - WHO WON?",
                              font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # System prediction
        sys_pred = validation_data['system_prediction']
        pred_label = tk.Label(root, text=f"System detected: {sys_pred} WINS",
                             font=('Arial', 12),
                             fg='blue' if sys_pred == 'P1' else 'red')
        pred_label.pack(pady=5)
        
        # Health info
        health_frame = tk.Frame(root)
        health_frame.pack(pady=5)
        
        tk.Label(health_frame, text=f"Health: P1={validation_data['p1_health']:.1f}% "
                                   f"P2={validation_data['p2_health']:.1f}%").pack()
        tk.Label(health_frame, text=f"First to zero: {validation_data['first_to_zero_flag']} "
                                   f"Both at zero: {validation_data['both_at_zero']}").pack()
        
        # Separator
        tk.Label(root, text="─" * 40).pack(pady=10)
        
        # Selection frame
        selection_frame = tk.Frame(root)
        selection_frame.pack(pady=10)
        
        tk.Label(selection_frame, text="Who ACTUALLY won?",
                font=('Arial', 12, 'bold')).pack()
        
        # Radio buttons
        radio_frame = tk.Frame(selection_frame)
        radio_frame.pack(pady=10)
        
        tk.Radiobutton(radio_frame, text="P1 WINS", variable=selected_winner,
                      value="P1", font=('Arial', 11)).pack(anchor='w')
        tk.Radiobutton(radio_frame, text="P2 WINS", variable=selected_winner,
                      value="P2", font=('Arial', 11)).pack(anchor='w')
        tk.Radiobutton(radio_frame, text="INVALID/NO WINNER", variable=selected_winner,
                      value="INVALID", font=('Arial', 11)).pack(anchor='w')
        
        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=15)
        
        def confirm():
            if selected_winner.get():
                result['confirmed'] = True
                result['winner'] = selected_winner.get()
                root.quit()
                root.destroy()
        
        def dismiss():
            result['confirmed'] = False
            root.quit()
            root.destroy()
        
        tk.Button(button_frame, text="✓ CONFIRM", command=confirm,
                 bg='green', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        tk.Button(button_frame, text="✗ DISMISS", command=dismiss,
                 bg='gray', fg='white', font=('Arial', 10)).pack(side='left', padx=5)
        
        # Keyboard shortcuts
        def on_key(event):
            if event.char == '1':
                selected_winner.set("P1")
                confirm()
            elif event.char == '2':
                selected_winner.set("P2")
                confirm()
            elif event.char.lower() == 'x':
                selected_winner.set("INVALID")
                confirm()
            elif event.keysym == 'Escape':
                dismiss()
        
        root.bind('<Key>', on_key)
        root.focus_set()
        
        # Instructions
        tk.Label(root, text="Shortcuts: 1=P1, 2=P2, X=Invalid, ESC=Dismiss",
                font=('Arial', 9), fg='gray').pack(pady=5)
        
        root.mainloop()
        return result
    
    def log_validation(self, validation_data, human_result):
        """Log the validation result to CSV"""
        system_correct = (validation_data['system_prediction'] == human_result.get('winner', ''))
        
        with open(self.validation_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                validation_data['timestamp'],
                validation_data['system_prediction'],
                human_result.get('winner', 'DISMISSED'),
                validation_data['p1_health'],
                validation_data['p2_health'],
                validation_data['first_to_zero_flag'],
                validation_data['both_at_zero'],
                system_correct
            ])
        
        if human_result.get('confirmed', False):
            winner = human_result['winner']
            if system_correct and winner != 'INVALID':
                log_round(f"✅ VALIDATION: Confirmed {winner} wins - System was CORRECT!")
            elif winner != 'INVALID':
                log_round(f"❌ VALIDATION: Human says {winner} wins - System was WRONG!")
            else:
                log_round(f"⚠️ VALIDATION: Human marked as INVALID round")
        else:
            log_round(f"⏭️ VALIDATION: Dismissed without confirmation")
    
    def run_gui_thread(self):
        """Run the GUI thread that processes validation requests"""
        while True:
            try:
                # Wait for validation request
                validation_data = self.validation_queue.get(timeout=1.0)
                
                # Show dialog and get result
                human_result = self.show_validation_dialog(validation_data)
                
                # Log the result
                self.log_validation(validation_data, human_result)
                
                # Put result in queue for main thread if needed
                self.result_queue.put({
                    'validation_data': validation_data,
                    'human_result': human_result
                })
                
            except:
                # Timeout - continue waiting
                continue