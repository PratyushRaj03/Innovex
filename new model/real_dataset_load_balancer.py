# real_dataset_load_balancer_fixed_stream.py
import numpy as np
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import Dict, List, Optional
import os
import glob
from collections import deque
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import codecs
import json
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper function to convert numpy types to Python native types
def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    else:
        return obj

class RealDatasetLoader:
    """Load real cloud workload datasets"""
    
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.data = None
        
    def load_local_dataset(self, file_path):
        """Load local CSV dataset"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None
    
    def preprocess_data(self, df, sample_size=5000):
        """Preprocess data to match our format"""
        if df is None:
            return None
        
        logger.info("Preprocessing data...")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            start_time = datetime(2024, 1, 1)
            df['timestamp'] = [start_time + timedelta(minutes=i) for i in range(len(df))]
        
        if 'vm_id' not in df.columns:
            df['vm_id'] = df.index % 8
        df['vm_id'] = df['vm_id'].astype(str)
        
        if 'cpu_util' in df.columns:
            df['cpu_util'] = pd.to_numeric(df['cpu_util'], errors='coerce').fillna(50)
        else:
            df['cpu_util'] = np.random.uniform(10, 90, len(df))
        
        if 'memory_util' in df.columns:
            df['memory_util'] = pd.to_numeric(df['memory_util'], errors='coerce').fillna(50)
        else:
            df['memory_util'] = df['cpu_util'] * 0.7 + np.random.normal(0, 5, len(df))
            df['memory_util'] = df['memory_util'].clip(0, 100)
        
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['load_score'] = 0.4 * (df['cpu_util'] / 100) + 0.3 * (df['memory_util'] / 100)
        
        # Add status based on load
        df['status'] = df['load_score'].apply(lambda x: 'Critical' if x > 0.8 else ('High' if x > 0.6 else ('Normal' if x > 0.3 else 'Idle')))
        
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).sort_values('timestamp')
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df


class OptimizedNeuralNetworkModels:
    """Optimized Neural Network models with consistent predictions"""
    
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_cols = None
        self.prediction_cache = {}
        
    def create_unified_model(self, input_dim):
        """Create a unified model that combines LSTM and FNN features"""
        lstm_input = layers.Input(shape=(self.sequence_length, input_dim), name='lstm_input')
        lstm_out = layers.LSTM(64, return_sequences=True)(lstm_input)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        lstm_out = layers.LSTM(32)(lstm_out)
        lstm_out = layers.Dropout(0.2)(lstm_out)
        
        fnn_input = layers.Input(shape=(input_dim,), name='fnn_input')
        fnn_out = layers.Dense(64, activation='relu')(fnn_input)
        fnn_out = layers.BatchNormalization()(fnn_out)
        fnn_out = layers.Dropout(0.3)(fnn_out)
        fnn_out = layers.Dense(32, activation='relu')(fnn_out)
        
        combined = layers.Concatenate()([lstm_out, fnn_out])
        combined = layers.Dense(32, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        combined = layers.Dense(16, activation='relu')(combined)
        output = layers.Dense(1)(combined)
        
        model = keras.Model(inputs=[lstm_input, fnn_input], outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), 
                     loss='mse', metrics=['mae'])
        return model
    
    def train_models(self, data, feature_cols):
        """Train optimized unified model"""
        logger.info("Training Optimized Unified Neural Network...")
        self.feature_cols = feature_cols
        
        X_lstm_list = []
        X_fnn_list = []
        y_list = []
        
        vms = data['vm_id'].unique()
        
        for vm_id in vms[:5]:
            vm_data = data[data['vm_id'] == vm_id].sort_values('timestamp')
            features = vm_data[feature_cols].values
            targets = vm_data['cpu_util'].values
            
            for i in range(len(features) - self.sequence_length):
                lstm_seq = features[i:i+self.sequence_length]
                fnn_current = features[i+self.sequence_length - 1]
                target = targets[i+self.sequence_length]
                
                X_lstm_list.append(lstm_seq)
                X_fnn_list.append(fnn_current)
                y_list.append(target)
        
        if len(X_lstm_list) < 50:
            logger.warning(f"Insufficient data: {len(X_lstm_list)} sequences")
            return {'unified': False}
        
        X_lstm = np.array(X_lstm_list)
        X_fnn = np.array(X_fnn_list)
        y = np.array(y_list)
        
        X_lstm_reshaped = X_lstm.reshape(-1, X_lstm.shape[-1])
        X_lstm_scaled = self.scaler.fit_transform(X_lstm_reshaped)
        X_lstm = X_lstm_scaled.reshape(X_lstm.shape)
        X_fnn_scaled = self.scaler.transform(X_fnn)
        
        split_idx = int(len(X_lstm) * 0.8)
        X_lstm_train = X_lstm[:split_idx]
        X_lstm_test = X_lstm[split_idx:]
        X_fnn_train = X_fnn_scaled[:split_idx]
        X_fnn_test = X_fnn_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        self.model = self.create_unified_model(len(feature_cols))
        
        logger.info(f"Training with {len(X_lstm_train)} sequences...")
        history = self.model.fit(
            [X_lstm_train, X_fnn_train], y_train,
            validation_data=([X_lstm_test, X_fnn_test], y_test),
            epochs=30,
            batch_size=32,
            verbose=1
        )
        
        loss, mae = self.model.evaluate([X_lstm_test, X_fnn_test], y_test, verbose=0)
        logger.info(f"Unified Model - Loss: {loss:.4f}, MAE: {mae:.4f}")
        
        self.is_trained = True
        return {'unified': True}
    
    def predict(self, recent_data):
        """Make consistent prediction using unified model"""
        if not self.is_trained or self.model is None:
            return None
        
        if len(recent_data) < self.sequence_length:
            return None
        
        try:
            cache_key = str(recent_data.iloc[-1]['timestamp'])
            
            if cache_key in self.prediction_cache:
                cache_time, cache_value = self.prediction_cache[cache_key]
                if (datetime.now() - cache_time).seconds < 2:
                    return cache_value
            
            lstm_features = recent_data[self.feature_cols].values[-self.sequence_length:]
            lstm_scaled = self.scaler.transform(lstm_features)
            X_lstm = lstm_scaled.reshape(1, self.sequence_length, len(self.feature_cols))
            
            current_features = recent_data.iloc[-1][self.feature_cols].values
            X_fnn = self.scaler.transform(current_features.reshape(1, -1))
            
            prediction = float(self.model.predict([X_lstm, X_fnn], verbose=0)[0][0])
            
            self.prediction_cache[cache_key] = (datetime.now(), prediction)
            
            current_time = datetime.now()
            old_keys = [k for k, (t, _) in self.prediction_cache.items() 
                       if (current_time - t).seconds > 5]
            for k in old_keys:
                del self.prediction_cache[k]
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def get_lstm_prediction(self, recent_data):
        if not self.is_trained or self.model is None or len(recent_data) < self.sequence_length:
            return None
        try:
            lstm_features = recent_data[self.feature_cols].values[-self.sequence_length:]
            lstm_scaled = self.scaler.transform(lstm_features)
            X_lstm = lstm_scaled.reshape(1, self.sequence_length, len(self.feature_cols))
            dummy_fnn = np.zeros((1, len(self.feature_cols)))
            return float(self.model.predict([X_lstm, dummy_fnn], verbose=0)[0][0])
        except:
            return None
    
    def get_fnn_prediction(self, recent_data):
        if not self.is_trained or self.model is None or len(recent_data) == 0:
            return None
        try:
            current_features = recent_data.iloc[-1][self.feature_cols].values
            X_fnn = self.scaler.transform(current_features.reshape(1, -1))
            dummy_lstm = np.zeros((1, self.sequence_length, len(self.feature_cols)))
            return float(self.model.predict([dummy_lstm, X_fnn], verbose=0)[0][0])
        except:
            return None


class RealTimeDatasetPlayer:
    """Play real dataset with manual loads"""
    
    def __init__(self, dataframe, timestamp_col='timestamp'):
        self.original_data = dataframe.sort_values(timestamp_col).reset_index(drop=True)
        self.timestamp_col = timestamp_col
        self.current_index = 0
        self.running = False
        self.playback_thread = None
        self.data_buffer = deque(maxlen=500)
        self.manual_loads = {}
        
    def start_playback(self):
        self.running = True
        self.current_index = 0
        self.data_buffer.clear()
        self.playback_thread = threading.Thread(target=self._playback_data)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        logger.info("Playback started")
        
    def stop_playback(self):
        self.running = False
        logger.info("Playback stopped")
    
    def clear_all_loads(self):
        self.manual_loads.clear()
        logger.info("All manual loads cleared")
        return True
    
    def add_manual_load(self, vm_id, cpu_increase, duration_minutes=5):
        vm_id_str = str(vm_id)
        self.manual_loads[vm_id_str] = {
            'vm_id': vm_id_str,
            'cpu_increase': float(cpu_increase),
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(minutes=duration_minutes),
            'active': True
        }
        logger.info(f"Manual load: VM {vm_id_str} +{cpu_increase}% CPU")
        return self.manual_loads[vm_id_str]
    
    def remove_expired_loads(self):
        current_time = datetime.now()
        expired = []
        for vm_id, load in self.manual_loads.items():
            if load['active'] and current_time >= load['end_time']:
                expired.append(vm_id)
        for vm_id in expired:
            del self.manual_loads[vm_id]
    
    def _apply_manual_loads(self, data_point):
        if not self.manual_loads:
            return data_point
        
        self.remove_expired_loads()
        
        if not self.manual_loads:
            return data_point
        
        modified = data_point.copy()
        vm_id_str = str(modified['vm_id'])
        
        if vm_id_str in self.manual_loads:
            load = self.manual_loads[vm_id_str]
            if load['active'] and datetime.now() < load['end_time']:
                modified['cpu_util'] = min(100, modified['cpu_util'] + load['cpu_increase'])
                modified['load_score'] = min(1.0, modified['load_score'] + load['cpu_increase'] / 100)
                modified['_manual_load'] = load['cpu_increase']
                # Update status
                if modified['load_score'] > 0.8:
                    modified['status'] = 'Critical (Manual Spike!)'
                elif modified['load_score'] > 0.6:
                    modified['status'] = 'High (Manual Spike)'
                else:
                    modified['status'] = 'Elevated'
        
        return modified
    
    def _playback_data(self):
        while self.running and self.current_index < len(self.original_data):
            original = self.original_data.iloc[self.current_index].copy()
            modified = self._apply_manual_loads(original)
            self.data_buffer.append(modified)
            self.current_index += 1
            
            if self.current_index < len(self.original_data):
                current_time = modified[self.timestamp_col]
                next_time = self.original_data.iloc[self.current_index][self.timestamp_col]
                time_diff = (next_time - current_time).total_seconds()
                if 0 < time_diff < 60:
                    time.sleep(time_diff)
                else:
                    time.sleep(0.1)
    
    def get_current_data(self):
        if len(self.data_buffer) > 0:
            return self.data_buffer[-1]
        return None
    
    def get_recent_data(self, n=100):
        if len(self.data_buffer) > 0:
            return pd.DataFrame(list(self.data_buffer)[-n:])
        return pd.DataFrame()
    
    def get_progress(self):
        if len(self.original_data) > 0:
            return float((self.current_index / len(self.original_data)) * 100)
        return 0.0
    
    def get_active_loads(self):
        self.remove_expired_loads()
        active = []
        for vm_id, load in self.manual_loads.items():
            if load['active']:
                remaining = max(0, (load['end_time'] - datetime.now()).total_seconds() / 60)
                active.append({
                    'vm_id': str(load['vm_id']),
                    'cpu_increase': float(load['cpu_increase']),
                    'remaining_minutes': float(round(remaining, 1))
                })
        return active


class RealDatasetLoadBalancer:
    """Main load balancer with optimized predictions"""
    
    def __init__(self):
        self.dataset_loader = RealDatasetLoader()
        self.dataset_player = None
        self.data = None
        self.nn_models = OptimizedNeuralNetworkModels()
        
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.streaming_active = False
        
        self.setup_routes()
        self.setup_socket_events()
        
    def load_dataset(self):
        logger.info("Loading dataset...")
        
        if os.path.exists('data/realistic_cloud_workload.csv'):
            self.data = self.dataset_loader.load_local_dataset('data/realistic_cloud_workload.csv')
        elif os.path.exists('data/'):
            csv_files = glob.glob('data/*.csv')
            if csv_files:
                self.data = self.dataset_loader.load_local_dataset(csv_files[0])
        
        if self.data is None:
            logger.info("Generating sample data...")
            self.data = self._generate_sample_data()
        
        if self.data is not None:
            self.data = self.dataset_loader.preprocess_data(self.data)
            self.dataset_player = RealTimeDatasetPlayer(self.data)
            logger.info(f"Loaded {len(self.data)} records")
            return True
        return False
    
    def _generate_sample_data(self):
        np.random.seed(42)
        n_records = 3000
        n_vms = 8
        data = []
        start_time = datetime(2024, 1, 1)
        
        for vm in range(n_vms):
            for i in range(n_records // n_vms):
                timestamp = start_time + timedelta(minutes=i * 15)
                hour = timestamp.hour
                day = timestamp.weekday()
                
                if 9 <= hour <= 17 and day < 5:
                    cpu = np.random.normal(65, 12)
                elif 18 <= hour <= 22:
                    cpu = np.random.normal(45, 10)
                else:
                    cpu = np.random.normal(25, 8)
                
                cpu = max(0, min(100, cpu + np.random.normal(0, 5)))
                memory = cpu * 0.8 + np.random.normal(0, 8)
                memory = max(0, min(100, memory))
                
                data.append({
                    'timestamp': timestamp,
                    'vm_id': str(vm),
                    'cpu_util': float(cpu),
                    'memory_util': float(memory),
                    'response_time': 50 + cpu * 2 + np.random.normal(0, 15)
                })
        
        df = pd.DataFrame(data)
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/generated_sample.csv', index=False)
        return df
    
    def train_models(self):
        if self.data is None:
            return False
        
        feature_cols = ['cpu_util', 'memory_util', 'hour_of_day', 'day_of_week']
        available_cols = [col for col in feature_cols if col in self.data.columns]
        return self.nn_models.train_models(self.data, available_cols)
    
    def setup_routes(self):
        
        @self.app.route('/')
        def index():
            return render_template('fixed_dashboard.html')
        
        @self.app.route('/api/stats')
        def get_stats():
            try:
                if self.data is not None:
                    vm_ids = sorted([str(v) for v in self.data['vm_id'].unique()])
                    return jsonify({
                        'total_records': int(len(self.data)),
                        'unique_vms': int(self.data['vm_id'].nunique()),
                        'vm_ids': vm_ids[:10],
                        'avg_cpu': float(self.data['cpu_util'].mean()),
                        'avg_memory': float(self.data['memory_util'].mean()),
                        'models_trained': bool(self.nn_models.is_trained)
                    })
                return jsonify({'status': 'error'}), 500
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/control/play', methods=['POST'])
        def start_playback():
            try:
                if self.dataset_player:
                    self.dataset_player.start_playback()
                    return jsonify({'status': 'success', 'message': 'Playback started'})
                return jsonify({'status': 'error'}), 500
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/control/stop', methods=['POST'])
        def stop_playback():
            try:
                if self.dataset_player:
                    self.dataset_player.stop_playback()
                    return jsonify({'status': 'success', 'message': 'Playback stopped'})
                return jsonify({'status': 'error'}), 500
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/predict', methods=['POST'])
        def predict():
            try:
                if not self.nn_models.is_trained:
                    return jsonify({'status': 'error', 'message': 'Models not trained'}), 400
                
                recent = self.dataset_player.get_recent_data(20) if self.dataset_player else None
                if recent is None or len(recent) < 10:
                    return jsonify({'status': 'error', 'message': 'Insufficient data'}), 400
                
                data = request.get_json() or {}
                model_type = data.get('model_type', 'unified')
                
                if model_type == 'unified':
                    prediction = self.nn_models.predict(recent)
                    lstm = self.nn_models.get_lstm_prediction(recent)
                    fnn = self.nn_models.get_fnn_prediction(recent)
                elif model_type == 'lstm':
                    prediction = self.nn_models.get_lstm_prediction(recent)
                    lstm = prediction
                    fnn = self.nn_models.get_fnn_prediction(recent)
                else:
                    prediction = self.nn_models.get_fnn_prediction(recent)
                    lstm = self.nn_models.get_lstm_prediction(recent)
                    fnn = prediction
                
                if prediction is not None:
                    current_cpu = float(recent.iloc[-1]['cpu_util']) if len(recent) > 0 else None
                    return jsonify({
                        'status': 'success',
                        'prediction': float(prediction),
                        'lstm': float(lstm) if lstm else None,
                        'fnn': float(fnn) if fnn else None,
                        'current_cpu': float(current_cpu) if current_cpu else None
                    })
                return jsonify({'status': 'error', 'message': 'Prediction failed'}), 400
            except Exception as e:
                logger.error(f"Predict error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/manual_load', methods=['POST'])
        def add_manual_load():
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'status': 'error', 'message': 'No data'}), 400
                
                vm_id = str(data.get('vm_id', '0'))
                cpu_increase = float(data.get('cpu_increase', 20))
                duration = int(data.get('duration', 5))
                
                if self.dataset_player:
                    load = self.dataset_player.add_manual_load(vm_id, cpu_increase, duration)
                    return jsonify({
                        'status': 'success',
                        'message': f'🔥 Added {cpu_increase}% load to VM {vm_id} for {duration} min',
                        'load': convert_to_native(load)
                    })
                return jsonify({'status': 'error'}), 500
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/active_loads')
        def get_active_loads():
            try:
                if self.dataset_player:
                    return jsonify({
                        'status': 'success',
                        'active_loads': convert_to_native(self.dataset_player.get_active_loads())
                    })
                return jsonify({'status': 'error', 'active_loads': []}), 500
            except Exception as e:
                return jsonify({'status': 'error', 'active_loads': []}), 500
        
        @self.app.route('/api/refresh')
        def refresh():
            try:
                if not self.dataset_player:
                    return jsonify({'status': 'error'}), 500
                
                self.dataset_player.clear_all_loads()
                
                current = self.dataset_player.get_current_data()
                recent = self.dataset_player.get_recent_data(20)
                
                unified_pred = self.nn_models.predict(recent) if len(recent) > 10 else None
                lstm_pred = self.nn_models.get_lstm_prediction(recent) if len(recent) > 10 else None
                fnn_pred = self.nn_models.get_fnn_prediction(recent) if len(recent) > 0 else None
                
                response = {
                    'status': 'success',
                    'current': None,
                    'predictions': {
                        'unified': float(unified_pred) if unified_pred else None,
                        'lstm': float(lstm_pred) if lstm_pred else None,
                        'fnn': float(fnn_pred) if fnn_pred else None
                    },
                    'active_loads': convert_to_native(self.dataset_player.get_active_loads()),
                    'progress': float(self.dataset_player.get_progress()),
                    'message': '🔄 Refreshed - All spikes cleared'
                }
                
                if current is not None:
                    response['current'] = {
                        'vm_id': str(current['vm_id']),
                        'cpu_util': float(current['cpu_util']),
                        'memory_util': float(current['memory_util']),
                        'load_score': float(current['load_score']),
                        'status': str(current.get('status', 'Normal')),
                        'timestamp': str(current['timestamp']),
                        'hour': int(current['hour_of_day']),
                        'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][int(current['day_of_week'])]
                    }
                
                return jsonify(convert_to_native(response))
            except Exception as e:
                logger.error(f"Refresh error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
    
    def setup_socket_events(self):
        
        @self.socketio.on('connect')
        def handle_connect():
            emit('connected', {'message': 'Connected'})
        
        @self.socketio.on('start_streaming')
        def handle_start_streaming():
            self.streaming_active = True
            threading.Thread(target=self._stream_data, daemon=True).start()
        
        @self.socketio.on('stop_streaming')
        def handle_stop_streaming():
            self.streaming_active = False
    
    def _stream_data(self):
        """Stream data to clients with proper type conversion"""
        while self.streaming_active:
            try:
                if self.dataset_player:
                    current = self.dataset_player.get_current_data()
                    if current is not None:
                        recent = self.dataset_player.get_recent_data(20)
                        
                        unified_pred = self.nn_models.predict(recent) if len(recent) > 10 else None
                        lstm_pred = self.nn_models.get_lstm_prediction(recent) if len(recent) > 10 else None
                        fnn_pred = self.nn_models.get_fnn_prediction(recent) if len(recent) > 0 else None
                        
                        # Convert all numpy types to Python native types
                        data = {
                            'timestamp': str(current['timestamp']),
                            'vm_id': str(current['vm_id']),
                            'cpu_util': float(current['cpu_util']),
                            'memory_util': float(current['memory_util']),
                            'load_score': float(current['load_score']),
                            'status': str(current.get('status', 'Normal')),
                            'hour': int(current['hour_of_day']),
                            'day': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][int(current['day_of_week'])],
                            'progress': float(self.dataset_player.get_progress()),
                            'active_loads': int(len(self.dataset_player.get_active_loads())),
                            'has_manual_load': bool('_manual_load' in current),
                            'manual_load_value': float(current.get('_manual_load', 0)),
                            'predictions': {
                                'unified': float(unified_pred) if unified_pred else None,
                                'lstm': float(lstm_pred) if lstm_pred else None,
                                'fnn': float(fnn_pred) if fnn_pred else None
                            }
                        }
                        
                        self.socketio.emit('data_update', convert_to_native(data))
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Stream error: {e}")
                time.sleep(1)
    
    def run(self, host='0.0.0.0', port=5000):
        print("\n" + "="*70)
        print("🚀 AI-DRIVEN LOAD BALANCER - Fixed JSON Serialization")
        print("="*70)
        
        print("\n📁 Loading dataset...")
        if not self.load_dataset():
            print("❌ Failed to load dataset")
            return
        
        print("\n🤖 Training Unified Model...")
        self.train_models()
        
        self._create_dashboard()
        
        print(f"\n🌐 Server: http://{host}:{port}")
        print("\n📊 FEATURES:")
        print("   1. UNIFIED MODEL: LSTM + FNN combined")
        print("   2. CONSISTENT PREDICTIONS: Same result every click")
        print("   3. FIXED JSON SERIALIZATION: No more int32 errors")
        print("   4. REAL-TIME STATUS: Critical/High/Normal/Idle indicators")
        print("   5. MANUAL LOAD: Visual spikes with duration tracking")
        print("   6. REFRESH: Clear spikes and update data")
        print("="*70)
        
        self.socketio.run(self.app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
    
    def _create_dashboard(self):
        os.makedirs('templates', exist_ok=True)
        
        dashboard_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Load Balancer</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .stat-value { font-size: 36px; font-weight: bold; color: #667eea; }
        .stat-label { color: #666; margin-top: 10px; }
        button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            margin: 5px;
            transition: all 0.3s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }
        .spike-badge {
            background: linear-gradient(135deg, #f44336, #ff9800);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            animation: pulse 1s infinite;
            display: inline-block;
        }
        .load-badge {
            background: linear-gradient(135deg, #ff9800, #f44336);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            margin: 8px 0;
        }
        .progress-bar {
            width: 100%;
            height: 35px;
            background: #e0e0e0;
            border-radius: 20px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .current-data-container {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            padding: 20px;
            border-left: 5px solid #667eea;
        }
        .current-data-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .data-item {
            background: white;
            padding: 12px;
            border-radius: 10px;
            text-align: center;
        }
        .data-label { font-size: 12px; color: #888; text-transform: uppercase; }
        .data-value { font-size: 20px; font-weight: bold; color: #667eea; margin-top: 5px; }
        .status-critical { color: #f44336; }
        .status-high { color: #ff9800; }
        .status-normal { color: #4caf50; }
        .status-idle { color: #9e9e9e; }
        .metric-value { font-size: 28px; font-weight: bold; color: #667eea; }
        .prediction-box {
            background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            margin-top: 15px;
        }
        .prediction-value { font-size: 56px; font-weight: bold; color: #4caf50; }
        .model-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 15px;
        }
        .model-card {
            background: #f5f5f5;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }
        .model-card h4 { margin-bottom: 10px; color: #667eea; }
        .model-prediction { font-size: 28px; font-weight: bold; margin: 10px 0; }
        .unified-card { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border: 2px solid #4caf50; }
        .manual-panel {
            background: linear-gradient(135deg, #fff3e0, #ffe0b2);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #ff9800;
        }
        input, select {
            padding: 10px;
            margin: 5px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
        }
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 24px;
            border-radius: 10px;
            z-index: 1000;
            animation: slideIn 0.3s;
            font-weight: 500;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.05); }
        }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .success { background: #4caf50; color: white; }
        .error { background: #f44336; color: white; }
        .warning { background: #ff9800; color: white; }
        .consistent-badge {
            background: #4caf50;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 10px;
            margin-left: 8px;
        }
        h3 { color: #333; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🚀 AI-Driven Dynamic Load Balancer</h1>
            <p><strong>Unified LSTM + FNN Model</strong> | Real-time Predictions | Visual Load Spikes</p>
            <div>
                <button onclick="startPlayback()">▶ Start Playback</button>
                <button onclick="stopPlayback()">⏸ Stop Playback</button>
                <button onclick="startStreaming()">📡 Start Streaming</button>
                <button onclick="stopStreaming()">🛑 Stop Streaming</button>
                <button onclick="refreshData()" style="background:#ff9800;">🔄 Refresh & Clear Spikes</button>
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value" id="total-records">-</div><div class="stat-label">Total Records</div></div>
            <div class="stat-card"><div class="stat-value" id="unique-vms">-</div><div class="stat-label">Virtual Machines</div></div>
            <div class="stat-card"><div class="stat-value" id="avg-cpu">-</div><div class="stat-label">Avg CPU Usage</div></div>
            <div class="stat-card"><div class="stat-value" id="avg-memory">-</div><div class="stat-label">Avg Memory Usage</div></div>
        </div>
        
        <div class="card">
            <h3>📈 Real-Time Metrics <span id="spike-indicator" style="display:none;" class="spike-badge">⚡ SPIKE ACTIVE!</span></h3>
            <canvas id="cpuChart" height="300"></canvas>
            <p style="text-align:center; margin-top:10px; color:#666;">
                <span style="color:#75c4c4;">● Actual CPU</span> 
                <span style="color:#9966ff;">● Memory Usage</span> 
                <span style="color:#4caf50;">● AI Prediction</span>
            </p>
        </div>
        
        <div class="card">
            <h3>📊 Playback Progress</h3>
            <div class="progress-bar"><div class="progress-fill" id="progress-fill">0%</div></div>
        </div>
        
        <div class="card">
            <h3>🖥️ Current VM Status</h3>
            <div id="current-data" class="current-data-container">
                <div style="text-align: center; color: #888;">Waiting for data...</div>
            </div>
        </div>
        
        <div class="card">
            <h3>🤖 AI Predictions (Unified Model)</h3>
            <div class="model-grid">
                <div class="model-card"><h4>📈 LSTM Component</h4><div class="model-prediction" id="lstm-pred">-</div><small>Time Series Analysis</small></div>
                <div class="model-card"><h4>🔢 FNN Component</h4><div class="model-prediction" id="fnn-pred">-</div><small>Feature Analysis</small></div>
                <div class="model-card unified-card"><h4>🎯 UNIFIED MODEL <span class="consistent-badge">CONSISTENT</span></h4><div class="model-prediction" id="unified-pred">-</div><small>Combined Intelligence</small></div>
            </div>
            <div style="text-align:center;">
                <button onclick="getPrediction('unified')">🎯 Get Unified Prediction</button>
                <button onclick="getPrediction('lstm')">📈 LSTM Only</button>
                <button onclick="getPrediction('fnn')">🔢 FNN Only</button>
            </div>
            <div id="prediction-result" style="display:none;" class="prediction-box"></div>
            <p style="text-align:center; margin-top:10px; font-size:12px; color:#4caf50;">✅ Same prediction every click - cached for consistency!</p>
        </div>
        
        <div class="card">
            <h3>⚡ Manual Load Injection</h3>
            <div class="manual-panel">
                <p><strong>Simulate Traffic Spike:</strong></p>
                <select id="vm-select"></select>
                <input type="number" id="cpu-inc" value="30" min="0" max="100" step="5">
                <input type="number" id="duration" value="2" min="1" max="10">
                <button onclick="addManualLoad()" style="background:#ff9800;">➕ Inject Load - Create Spike!</button>
            </div>
            <div id="active-loads" style="margin-top:15px;"></div>
        </div>
    </div>
    
    <script>
        let socket = null;
        let chart = null;
        let history = [];
        
        function initChart() {
            const ctx = document.getElementById('cpuChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        { label: 'CPU Usage (%)', data: [], borderColor: 'rgb(75,192,192)', backgroundColor: 'rgba(75,192,192,0.1)', fill: true, tension: 0.1, borderWidth: 3 },
                        { label: 'Memory Usage (%)', data: [], borderColor: 'rgb(153,102,255)', backgroundColor: 'rgba(153,102,255,0.1)', fill: true, tension: 0.1 },
                        { label: 'AI Prediction', data: [], borderColor: 'rgb(76,175,80)', backgroundColor: 'rgba(76,175,80,0.1)', borderDash: [8, 4], fill: false, tension: 0.1, borderWidth: 2 }
                    ]
                },
                options: { 
                    responsive: true, 
                    maintainAspectRatio: true, 
                    scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: 'Utilization (%)' } } }
                }
            });
        }
        
        function connectSocket() {
            socket = io();
            socket.on('connect', () => showMessage('✅ Connected to AI Load Balancer', 'success'));
            socket.on('data_update', (data) => updateDisplay(data));
            socket.on('disconnect', () => showMessage('❌ Disconnected', 'error'));
        }
        
        function updateDisplay(data) {
            document.getElementById('progress-fill').style.width = data.progress + '%';
            document.getElementById('progress-fill').innerHTML = data.progress.toFixed(1) + '%';
            
            let statusClass = '';
            if (data.status.includes('Critical')) statusClass = 'status-critical';
            else if (data.status.includes('High')) statusClass = 'status-high';
            else if (data.status.includes('Normal')) statusClass = 'status-normal';
            else if (data.status.includes('Idle')) statusClass = 'status-idle';
            
            let spikeHtml = '';
            if (data.has_manual_load) {
                document.getElementById('spike-indicator').style.display = 'inline-block';
                spikeHtml = `<div style="background: linear-gradient(135deg, #f44336, #ff9800); color: white; padding: 10px; border-radius: 10px; margin-top: 10px; text-align: center;">⚡ MANUAL SPIKE ACTIVE! +${data.manual_load_value}% CPU Added</div>`;
            } else {
                document.getElementById('spike-indicator').style.display = 'none';
            }
            
            document.getElementById('current-data').innerHTML = `
                <div class="current-data-grid">
                    <div class="data-item"><div class="data-label">🖥️ VM</div><div class="data-value">${data.vm_id}</div></div>
                    <div class="data-item"><div class="data-label">📊 Status</div><div class="data-value ${statusClass}">${data.status}</div></div>
                    <div class="data-item"><div class="data-label">⚙️ CPU</div><div class="data-value" style="color: ${data.has_manual_load ? '#f44336' : '#667eea'}">${data.cpu_util.toFixed(1)}%</div></div>
                    <div class="data-item"><div class="data-label">💾 Memory</div><div class="data-value">${data.memory_util.toFixed(1)}%</div></div>
                    <div class="data-item"><div class="data-label">📈 Load Score</div><div class="data-value">${data.load_score.toFixed(3)}</div></div>
                    <div class="data-item"><div class="data-label">📅 Day/Hour</div><div class="data-value">${data.day}, ${data.hour}:00</div></div>
                    <div class="data-item"><div class="data-label">🎯 Active Spikes</div><div class="data-value">${data.active_loads}</div></div>
                </div>
                ${spikeHtml}
            `;
            
            if (data.predictions.lstm) document.getElementById('lstm-pred').innerHTML = data.predictions.lstm.toFixed(1) + '%';
            if (data.predictions.fnn) document.getElementById('fnn-pred').innerHTML = data.predictions.fnn.toFixed(1) + '%';
            if (data.predictions.unified) document.getElementById('unified-pred').innerHTML = data.predictions.unified.toFixed(1) + '%';
            
            const time = new Date(data.timestamp).toLocaleTimeString();
            history.push({ time, cpu: data.cpu_util, memory: data.memory_util, pred: data.predictions.unified });
            if (history.length > 100) history.shift();
            
            chart.data.labels = history.map(d => d.time);
            chart.data.datasets[0].data = history.map(d => d.cpu);
            chart.data.datasets[1].data = history.map(d => d.memory);
            chart.data.datasets[2].data = history.map(d => d.pred);
            chart.update();
        }
        
        function showMessage(msg, type) {
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.innerHTML = msg;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }
        
        function startPlayback() { fetch('/api/control/play', {method:'POST'}).then(r=>r.json()).then(d=>showMessage(d.message,'success')); }
        function stopPlayback() { fetch('/api/control/stop', {method:'POST'}).then(r=>r.json()).then(d=>showMessage(d.message,'success')); }
        function startStreaming() { if(socket) socket.emit('start_streaming'); showMessage('📡 Streaming started', 'success'); }
        function stopStreaming() { if(socket) socket.emit('stop_streaming'); showMessage('🛑 Streaming stopped', 'info'); }
        function refreshData() { fetch('/api/refresh').then(r=>r.json()).then(d=>showMessage(d.message,'success')); }
        
        async function getPrediction(type) {
            const btn = event.target;
            btn.disabled = true;
            const originalText = btn.textContent;
            btn.textContent = '🤖 Predicting...';
            try {
                const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({model_type: type}) });
                const data = await response.json();
                if (data.status === 'success') {
                    const resultDiv = document.getElementById('prediction-result');
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = `<div style="font-size:18px;">${type.toUpperCase()}</div><div class="prediction-value">${data.prediction.toFixed(1)}%</div><small>Current CPU: ${data.current_cpu?.toFixed(1) || '-'}%</small>`;
                    showMessage(`${type} Prediction: ${data.prediction.toFixed(1)}%`, 'success');
                } else { showMessage(data.message, 'error'); }
            } catch(e) { showMessage('Error: ' + e.message, 'error'); }
            finally { btn.disabled = false; btn.textContent = originalText; }
        }
        
        async function addManualLoad() {
            const vm = document.getElementById('vm-select').value;
            const inc = parseFloat(document.getElementById('cpu-inc').value);
            const dur = parseInt(document.getElementById('duration').value);
            showMessage(`🔥 Injecting ${inc}% load to VM ${vm}`, 'warning');
            try {
                const response = await fetch('/api/manual_load', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({vm_id: vm, cpu_increase: inc, duration: dur}) });
                const data = await response.json();
                if (data.status === 'success') { showMessage(data.message, 'success'); setTimeout(getActiveLoads, 500); }
                else { showMessage(data.message, 'error'); }
            } catch(e) { showMessage('Error: ' + e.message, 'error'); }
        }
        
        async function getActiveLoads() {
            try {
                const response = await fetch('/api/active_loads');
                const data = await response.json();
                if (data.status === 'success' && data.active_loads.length > 0) {
                    let html = '<strong>🔥 Active Spikes:</strong><br>';
                    data.active_loads.forEach(l => { html += `<div class="load-badge">⚡ VM ${l.vm_id}: +${l.cpu_increase}% CPU (${l.remaining_minutes} min left)</div>`; });
                    document.getElementById('active-loads').innerHTML = html;
                } else { document.getElementById('active-loads').innerHTML = '✅ No active spikes'; }
            } catch(e) { console.error(e); }
        }
        
        async function loadVMList() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                if (data.vm_ids) {
                    const select = document.getElementById('vm-select');
                    select.innerHTML = '';
                    data.vm_ids.forEach(vm => { const option = document.createElement('option'); option.value = vm; option.textContent = `VM ${vm}`; select.appendChild(option); });
                }
            } catch(e) { console.error(e); }
        }
        
        fetch('/api/stats').then(r=>r.json()).then(data => {
            if (data.total_records) {
                document.getElementById('total-records').innerHTML = data.total_records;
                document.getElementById('unique-vms').innerHTML = data.unique_vms;
                document.getElementById('avg-cpu').innerHTML = data.avg_cpu?.toFixed(1) + '%';
                document.getElementById('avg-memory').innerHTML = data.avg_memory?.toFixed(1) + '%';
            }
        });
        
        initChart();
        connectSocket();
        loadVMList();
        setInterval(getActiveLoads, 3000);
    </script>
</body>
</html>"""
        
        with codecs.open('templates/fixed_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        logger.info("Fixed dashboard created")


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    balancer = RealDatasetLoadBalancer()
    try:
        balancer.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        traceback.print_exc()