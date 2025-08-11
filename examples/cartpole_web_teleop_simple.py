#!/usr/bin/env python3
"""
CartPole Web-Based Teleoperation Demo with Configurable Latency (Simple Version)

This demo provides a web interface for teleoperating the CartPole environment
with configurable network latency. Uses Gymnasium directly with latency simulation.

Usage:
    python cartpole_web_teleop_simple.py --latency 100 --port 5000
    Then open http://localhost:5000 in your browser
"""

import numpy as np
import time
import argparse
import json
import threading
import queue
from collections import deque
from dataclasses import dataclass
import gymnasium as gym

from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>CartPole Teleoperation with Latency</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .game-area {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        #canvas {
            display: block;
            margin: 0 auto;
            border: 3px solid #764ba2;
            border-radius: 10px;
            background: white;
        }
        .controls {
            margin-top: 20px;
            text-align: center;
            display: flex;
            justify-content: center;
            gap: 10px;
        }
        button {
            padding: 15px 30px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .action-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .action-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .action-btn:active, .action-btn.active {
            transform: translateY(0);
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        .reset-btn {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        .stats {
            margin-top: 20px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            color: #333;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-item {
            padding: 15px;
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .config {
            margin-bottom: 20px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            color: #333;
        }
        .config h2 {
            margin-top: 0;
            color: #667eea;
        }
        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .config-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .config-label {
            flex: 1;
            font-weight: 500;
        }
        .config-input {
            width: 100px;
            padding: 8px;
            border: 2px solid #667eea;
            border-radius: 5px;
            font-size: 16px;
        }
        .config-button {
            margin-top: 15px;
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        .latency-indicator {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            color: #333;
        }
        .latency-bar {
            height: 30px;
            background: #f0f0f0;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
            margin-top: 10px;
        }
        .latency-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
        }
        .message {
            margin: 20px auto;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            max-width: 1200px;
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .success {
            background: #4CAF50;
            color: white;
        }
        .failure {
            background: #F44336;
            color: white;
        }
        .keyboard-hint {
            margin-top: 15px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            text-align: center;
        }
    </style>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>🎮 CartPole Teleoperation with Network Latency</h1>
        
        <div id="message" class="message" style="display: none;"></div>
        
        <div class="config">
            <h2>⚙️ Configuration</h2>
            <div class="config-grid">
                <div class="config-item">
                    <label class="config-label">Latency (ms):</label>
                    <input type="number" id="latency" class="config-input" value="{{ latency_ms }}" min="0" max="2000" step="10">
                </div>
                <div class="config-item">
                    <label class="config-label">FPS:</label>
                    <input type="number" id="fps" class="config-input" value="30" min="10" max="60" step="5">
                </div>
            </div>
            <button class="config-button" onclick="applyConfig()">Apply Configuration</button>
        </div>
        
        <div class="game-area">
            <canvas id="canvas" width="800" height="400"></canvas>
            
            <div class="controls">
                <button class="action-btn" id="left-btn" onmousedown="sendAction(0)" onmouseup="stopAction()">
                    ⬅️ LEFT (A)
                </button>
                <button class="reset-btn" onclick="resetEnv()">
                    🔄 RESET (R)
                </button>
                <button class="action-btn" id="right-btn" onmousedown="sendAction(1)" onmouseup="stopAction()">
                    RIGHT (D) ➡️
                </button>
            </div>
            
            <div class="keyboard-hint">
                💡 Use keyboard: A/← for Left, D/→ for Right, R to Reset
            </div>
        </div>
        
        <div class="latency-indicator">
            <div style="display: flex; justify-content: space-between;">
                <span>Simulated Network Latency</span>
                <span id="latency-value">0 ms</span>
            </div>
            <div class="latency-bar">
                <div class="latency-fill" id="latency-fill" style="width: 0%">
                    <span id="latency-bar-text">0 ms</span>
                </div>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-label">Episode Steps</div>
                <div class="stat-value" id="episode-steps">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total Reward</div>
                <div class="stat-value" id="total-reward">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Cart Position</div>
                <div class="stat-value" id="cart-pos">0.00</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Pole Angle</div>
                <div class="stat-value" id="pole-angle">0.0°</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Cart Velocity</div>
                <div class="stat-value" id="cart-vel">0.00</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Pole Angular Vel</div>
                <div class="stat-value" id="pole-vel">0.00</div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let currentAction = null;
        let isConnected = false;
        let episodeActive = false;
        
        // Connect to server
        socket.on('connect', function() {
            console.log('Connected to server');
            isConnected = true;
            resetEnv();
        });
        
        // Handle state updates
        socket.on('state_update', function(data) {
            updateDisplay(data);
        });
        
        // Handle episode end
        socket.on('episode_end', function(data) {
            episodeActive = false;
            showMessage(data.message, data.success ? 'success' : 'failure');
        });
        
        // Handle reset confirmation
        socket.on('reset_complete', function(data) {
            episodeActive = true;
            hideMessage();
            updateDisplay(data);
        });
        
        // Keyboard controls
        document.addEventListener('keydown', function(e) {
            if (e.repeat) return;
            
            switch(e.key.toLowerCase()) {
                case 'a':
                case 'arrowleft':
                    e.preventDefault();
                    sendAction(0);
                    document.getElementById('left-btn').classList.add('active');
                    break;
                case 'd':
                case 'arrowright':
                    e.preventDefault();
                    sendAction(1);
                    document.getElementById('right-btn').classList.add('active');
                    break;
                case 'r':
                    e.preventDefault();
                    resetEnv();
                    break;
            }
        });
        
        document.addEventListener('keyup', function(e) {
            switch(e.key.toLowerCase()) {
                case 'a':
                case 'arrowleft':
                    document.getElementById('left-btn').classList.remove('active');
                    break;
                case 'd':
                case 'arrowright':
                    document.getElementById('right-btn').classList.remove('active');
                    break;
            }
        });
        
        function sendAction(action) {
            if (!isConnected) return;
            // Allow sending actions even after episode ends (to see the fallen state)
            currentAction = action;
            socket.emit('action', {action: action});
        }
        
        function stopAction() {
            currentAction = null;
        }
        
        function resetEnv() {
            if (!isConnected) return;
            socket.emit('reset');
            hideMessage();
        }
        
        function applyConfig() {
            const latency = document.getElementById('latency').value;
            const fps = document.getElementById('fps').value;
            
            socket.emit('update_config', {
                latency_ms: parseFloat(latency),
                fps: parseInt(fps)
            });
            
            showMessage('Configuration updated!', 'info');
            setTimeout(hideMessage, 2000);
            resetEnv();
        }
        
        function updateDisplay(data) {
            // Update stats
            document.getElementById('episode-steps').textContent = data.episode_steps;
            document.getElementById('total-reward').textContent = data.total_reward.toFixed(1);
            document.getElementById('cart-pos').textContent = data.cart_pos.toFixed(3);
            document.getElementById('pole-angle').textContent = (data.pole_angle * 180 / Math.PI).toFixed(1) + '°';
            document.getElementById('cart-vel').textContent = data.cart_vel.toFixed(3);
            document.getElementById('pole-vel').textContent = data.pole_vel.toFixed(3);
            
            // Update latency display
            const latencyMs = data.latency_ms;
            document.getElementById('latency-value').textContent = latencyMs.toFixed(1) + ' ms';
            document.getElementById('latency-bar-text').textContent = latencyMs.toFixed(0) + ' ms';
            
            // Update latency bar (max 500ms for visualization)
            const latencyPercent = Math.min(100, (latencyMs / 500) * 100);
            document.getElementById('latency-fill').style.width = latencyPercent + '%';
            
            // Draw the CartPole
            drawCartPole(data);
        }
        
        function drawCartPole(data) {
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const centerX = canvas.width / 2;
            const groundY = canvas.height * 0.7;
            const scale = 100;
            
            // Draw ground
            ctx.strokeStyle = '#666';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, groundY);
            ctx.lineTo(canvas.width, groundY);
            ctx.stroke();
            
            // Cart position (scaled)
            const cartX = centerX + data.cart_pos * scale;
            const cartWidth = 50;
            const cartHeight = 30;
            
            // Draw cart
            ctx.fillStyle = '#764ba2';
            ctx.fillRect(cartX - cartWidth/2, groundY - cartHeight, cartWidth, cartHeight);
            
            // Draw wheels
            ctx.fillStyle = '#333';
            ctx.beginPath();
            ctx.arc(cartX - cartWidth/3, groundY, 8, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.arc(cartX + cartWidth/3, groundY, 8, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw pole
            const poleLength = 100;
            const poleEndX = cartX + Math.sin(data.pole_angle) * poleLength;
            const poleEndY = groundY - cartHeight - Math.cos(data.pole_angle) * poleLength;
            
            ctx.strokeStyle = '#667eea';
            ctx.lineWidth = 8;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(cartX, groundY - cartHeight);
            ctx.lineTo(poleEndX, poleEndY);
            ctx.stroke();
            
            // Draw pole tip
            ctx.fillStyle = '#f093fb';
            ctx.beginPath();
            ctx.arc(poleEndX, poleEndY, 10, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw action indicator
            if (data.current_action !== null) {
                ctx.fillStyle = data.current_action === 0 ? '#4CAF50' : '#2196F3';
                ctx.font = 'bold 30px Arial';
                ctx.textAlign = 'center';
                const arrow = data.current_action === 0 ? '⬅' : '➡';
                ctx.fillText(arrow, cartX, groundY - cartHeight - 20);
            }
            
            // Draw delayed action indicator (what the cart sees)
            if (data.delayed_action !== null && data.delayed_action !== data.current_action) {
                ctx.fillStyle = 'rgba(255, 165, 0, 0.5)';
                ctx.font = 'bold 25px Arial';
                ctx.textAlign = 'center';
                const delayedArrow = data.delayed_action === 0 ? '⬅' : '➡';
                ctx.fillText(delayedArrow, cartX, groundY + 40);
                ctx.font = '12px Arial';
                ctx.fillText('(delayed)', cartX, groundY + 55);
            }
            
            // Draw position limits
            const limitX = 2.4 * scale;
            ctx.strokeStyle = '#F44336';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(centerX - limitX, 0);
            ctx.lineTo(centerX - limitX, canvas.height);
            ctx.moveTo(centerX + limitX, 0);
            ctx.lineTo(centerX + limitX, canvas.height);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        
        function showMessage(text, type) {
            const messageEl = document.getElementById('message');
            messageEl.textContent = text;
            messageEl.className = 'message ' + type;
            messageEl.style.display = 'block';
        }
        
        function hideMessage() {
            document.getElementById('message').style.display = 'none';
        }
    </script>
</body>
</html>
"""


class CartPoleWebTeleop:
    """Web-based teleoperation for CartPole with latency simulation."""
    
    def __init__(self, latency_ms=0, fps=30):
        self.latency_ms = latency_ms
        self.fps = fps
        self.timestep = 1.0 / fps
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'cartpole-teleop-secret'
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Create Gymnasium environment
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.obs = None
        self.episode_steps = 0
        self.total_reward = 0.0
        self.done = False
        
        # Action buffer for latency simulation
        self.action_buffer = deque()
        self.current_action = None
        self.last_executed_action = None
        
        # Background thread for environment updates
        self.running = False
        self.update_thread = None
        
        self.setup_routes()
        self.setup_socket_handlers()
        
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(
                HTML_TEMPLATE,
                latency_ms=self.latency_ms
            )
    
    def setup_socket_handlers(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')
            self.start_update_loop()
            self.reset_environment()
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')
            self.stop_update_loop()
        
        @self.socketio.on('action')
        def handle_action(data):
            if not self.done:
                action = data['action']
                # Add action to buffer with timestamp
                self.action_buffer.append((action, time.time()))
                self.current_action = action
        
        @self.socketio.on('reset')
        def handle_reset():
            self.reset_environment()
        
        @self.socketio.on('update_config')
        def handle_config_update(data):
            if 'latency_ms' in data:
                self.latency_ms = data['latency_ms']
            if 'fps' in data:
                self.fps = data['fps']
                self.timestep = 1.0 / data['fps']
            self.reset_environment()
    
    def reset_environment(self):
        """Reset the environment."""
        self.obs, _ = self.env.reset()
        self.episode_steps = 0
        self.total_reward = 0.0
        self.done = False
        self.action_buffer.clear()
        self.current_action = None
        self.last_executed_action = None
        
        # Send initial state
        self.socketio.emit('reset_complete', self.get_state_data())
    
    def get_delayed_action(self):
        """Get the action to execute considering delay."""
        if self.latency_ms <= 0 or len(self.action_buffer) == 0:
            return self.current_action if self.current_action is not None else 0
        
        # Clean old actions from buffer
        current_time = time.time()
        delay_seconds = self.latency_ms / 1000.0
        
        # Remove actions older than 2x delay
        while self.action_buffer and current_time - self.action_buffer[0][1] > delay_seconds * 2:
            self.action_buffer.popleft()
        
        # Find action from ~delay seconds ago
        delayed_action = None
        for action, timestamp in self.action_buffer:
            if current_time - timestamp >= delay_seconds:
                delayed_action = action
                break
        
        # If no delayed action found, use the oldest action or default
        if delayed_action is None:
            if self.action_buffer:
                delayed_action = self.action_buffer[0][0]
            else:
                delayed_action = 0
        
        return delayed_action
    
    def update_environment(self):
        """Update the environment in real-time."""
        while self.running:
            start_time = time.time()
            
            # Make sure environment is initialized
            if self.obs is None:
                time.sleep(0.1)
                continue
            
            # Always step the environment, even after "done"
            # Get action with delay
            action_to_execute = self.get_delayed_action()
            self.last_executed_action = action_to_execute
            
            # Step environment
            if not self.done:
                try:
                    self.obs, reward, terminated, truncated, _ = self.env.step(action_to_execute)
                    self.episode_steps += 1
                    self.total_reward += reward
                    
                    # Check if episode ended but don't stop updating
                    if (terminated or truncated) and not self.done:
                        self.done = True
                        success = self.episode_steps >= 195
                        message = f"Episode ended after {self.episode_steps} steps! "
                        if success:
                            message += "🎉 Great job balancing!"
                        else:
                            message += "💥 Pole fell! Press R to reset."
                        
                        self.socketio.emit('episode_end', {
                            'success': success,
                            'message': message,
                            'steps': self.episode_steps,
                            'reward': self.total_reward
                        })
                except Exception as e:
                    print(f"Error stepping environment: {e}")
                    # Environment needs reset
                    pass
            else:
                # Continue to show the fallen state
                # Just send the current state without stepping
                pass
            
            # Always send state update
            state_data = self.get_state_data()
            if state_data:  # Only emit if we have valid state data
                self.socketio.emit('state_update', state_data)
            
            # Sleep to maintain frame rate
            elapsed = time.time() - start_time
            sleep_time = max(0, self.timestep - elapsed)
            time.sleep(sleep_time)
    
    def start_update_loop(self):
        """Start the background update thread."""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self.update_environment, daemon=True)
            self.update_thread.start()
    
    def stop_update_loop(self):
        """Stop the background update thread."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
    
    def get_state_data(self):
        """Get current state data for client."""
        if self.obs is None:
            return {}
        
        return {
            'cart_pos': float(self.obs[0]),
            'cart_vel': float(self.obs[1]),
            'pole_angle': float(self.obs[2]),
            'pole_vel': float(self.obs[3]),
            'episode_steps': self.episode_steps,
            'total_reward': self.total_reward,
            'done': self.done,
            'latency_ms': self.latency_ms,
            'current_action': self.current_action,
            'delayed_action': self.last_executed_action
        }
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web server."""
        print(f"Starting CartPole Web Teleoperation Server...")
        print(f"Configuration:")
        print(f"  - Latency: {self.latency_ms} ms")
        print(f"  - FPS: {self.fps}")
        print(f"\nOpen http://localhost:{port} in your browser")
        print("\nTry different latencies to see the effect:")
        print("  - 0ms: No delay (baseline)")
        print("  - 100ms: Noticeable delay")
        print("  - 300ms: Challenging!")
        print("  - 500ms: Very difficult!\n")
        
        self.socketio.run(self.app, host=host, port=port, debug=debug)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CartPole Web Teleoperation with Configurable Latency (Simple Version)"
    )
    
    parser.add_argument(
        "--latency",
        type=float,
        default=0,
        help="Initial network latency in milliseconds (default: 0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5020,
        help="Web server port (default: 5000)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Web server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Create and run the web teleoperation
    teleop = CartPoleWebTeleop(latency_ms=args.latency, fps=args.fps)
    teleop.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()