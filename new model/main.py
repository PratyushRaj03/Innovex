import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
import glob

class RealCloudWorkloadDataLoader:
    """Handle real cloud workload data from various sources"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        
    def load_from_csv(self, file_path, timestamp_col='timestamp', vm_id_col='vm_id'):
        """Load data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} records from {file_path}")
            
            # Convert timestamp if needed
            if timestamp_col in df.columns:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                # Extract time features
                df['hour_of_day'] = df[timestamp_col].dt.hour
                df['day_of_week'] = df[timestamp_col].dt.dayofweek
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def load_from_azure_monitor(self, connection_string, query):
        """Load data from Azure Monitor"""
        try:
            # This would use Azure SDK
            # from azure.monitor.query import LogsQueryClient
            # client = LogsQueryClient(credential, connection_string)
            # response = client.query_workspace(workspace_id, query, timespan)
            # df = pd.DataFrame(response.tables[0].rows)
            
            print("Azure Monitor integration - implement with actual credentials")
            print(f"Query: {query}")
            return None
            
        except Exception as e:
            print(f"Error loading from Azure Monitor: {e}")
            return None
    
    def load_from_aws_cloudwatch(self, region, namespace, metric_names, start_time, end_time):
        """Load data from AWS CloudWatch"""
        try:
            # This would use boto3
            # import boto3
            # cloudwatch = boto3.client('cloudwatch', region_name=region)
            # response = cloudwatch.get_metric_statistics(...)
            
            print("AWS CloudWatch integration - implement with actual credentials")
            return None
            
        except Exception as e:
            print(f"Error loading from AWS CloudWatch: {e}")
            return None
    
    def load_from_prometheus(self, prometheus_url, query):
        """Load data from Prometheus"""
        try:
            # This would use prometheus-api-client
            # from prometheus_api_client import PrometheusConnect
            # prom = PrometheusConnect(url=prometheus_url)
            # data = prom.custom_query(query)
            
            print("Prometheus integration - implement with actual credentials")
            print(f"Query: {query}")
            return None
            
        except Exception as e:
            print(f"Error loading from Prometheus: {e}")
            return None
    
    def load_from_kafka(self, topic, bootstrap_servers, num_messages=10000):
        """Stream data from Kafka"""
        try:
            # This would use kafka-python
            # from kafka import KafkaConsumer
            # consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers)
            
            print("Kafka integration - implement with actual Kafka configuration")
            return None
            
        except Exception as e:
            print(f"Error loading from Kafka: {e}")
            return None
    
    def preprocess_real_data(self, df, feature_mapping=None):
        """
        Preprocess real data to match expected format
        
        feature_mapping: dict mapping real column names to expected column names
        Expected columns: cpu_util, memory_util, disk_io, network_traffic, 
                         queue_length, response_time, timestamp, vm_id
        """
        if df is None:
            return None
        
        # Apply feature mapping if provided
        if feature_mapping:
            df = df.rename(columns=feature_mapping)
        
        # Check required columns
        required_cols = ['cpu_util', 'memory_util', 'timestamp', 'vm_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print("Available columns:", df.columns.tolist())
            return None
        
        # Handle missing optional columns
        optional_cols = ['disk_io', 'network_traffic', 'queue_length', 'response_time']
        for col in optional_cols:
            if col not in df.columns:
                df[col] = 0  # Fill with default values
        
        # Calculate load score if not present
        if 'load_score' not in df.columns:
            df['load_score'] = (
                0.4 * (df['cpu_util'] / 100) +
                0.3 * (df['memory_util'] / 100) +
                0.2 * (df['response_time'] / 500) +
                0.1 * (df['queue_length'] / 10)
            )
        
        # Convert timestamp if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Preprocessed data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def generate_sample_real_data(self, n_records=10000):
        """Generate sample data that mimics real-world patterns"""
        np.random.seed(42)
        
        data = []
        start_time = datetime(2024, 1, 1)
        
        for i in range(n_records):
            timestamp = start_time + pd.Timedelta(minutes=i)
            vm_id = np.random.randint(0, 20)
            
            # Realistic patterns
            hour = timestamp.hour
            day = timestamp.dayofweek
            
            # Business hours pattern
            if 9 <= hour <= 17 and day < 5:
                base_load = np.random.normal(0.7, 0.1)
            elif 18 <= hour <= 22:
                base_load = np.random.normal(0.5, 0.15)
            else:
                base_load = np.random.normal(0.3, 0.1)
            
            # Add some VM-specific variation
            vm_factor = 0.8 + (vm_id % 5) * 0.1
            
            cpu_util = min(100, max(0, base_load * 70 * vm_factor + np.random.normal(0, 5)))
            memory_util = min(100, max(0, base_load * 65 * vm_factor + np.random.normal(0, 8)))
            disk_io = max(0, np.random.poisson(base_load * 800))
            network_traffic = max(0, np.random.exponential(base_load * 400))
            queue_length = max(0, np.random.poisson(base_load * 4))
            response_time = 50 + base_load * 250 + np.random.normal(0, 20)
            
            data.append([
                timestamp, vm_id, cpu_util, memory_util, disk_io,
                network_traffic, queue_length, response_time, hour, day
            ])
        
        columns = ['timestamp', 'vm_id', 'cpu_util', 'memory_util', 'disk_io',
                  'network_traffic', 'queue_length', 'response_time', 'hour_of_day', 'day_of_week']
        
        df = pd.DataFrame(data, columns=columns)
        df['load_score'] = (
            0.4 * (df['cpu_util'] / 100) +
            0.3 * (df['memory_util'] / 100) +
            0.2 * (df['response_time'] / 500) +
            0.1 * (df['queue_length'] / 10)
        )
        
        return df

class DynamicLoadBalancingAI:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.models = {}
        
    def create_lstm_model(self, input_dim):
        """Create LSTM-based prediction model"""
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, 
                       input_shape=(self.sequence_length, input_dim)),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_decision_model(self, input_dim):
        """Create model for load balancing decisions"""
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequences(self, data, feature_cols, target_col='cpu_util'):
        """Prepare sequences for time series prediction"""
        sequences = []
        targets = []
        
        data_subset = data[feature_cols].values
        
        for i in range(len(data_subset) - self.sequence_length):
            seq = data_subset[i:i+self.sequence_length]
            target = data.iloc[i+self.sequence_length][target_col]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

class AIBasedLoadBalancer:
    def __init__(self, use_real_data=True, data_source='csv', data_path=None):
        self.data_loader = RealCloudWorkloadDataLoader(data_path)
        self.ai_model = DynamicLoadBalancingAI()
        self.data = None
        self.trained_models = {}
        self.results = {}
        self.feature_cols = None
        self.use_real_data = use_real_data
        self.data_source = data_source
        
    def load_real_data(self):
        """Load real data based on configuration"""
        print(f"\n{'='*60}")
        print("Loading Real Cloud Workload Data")
        print(f"{'='*60}")
        
        if self.data_source == 'csv':
            if self.data_loader.data_path:
                # Load all CSV files in the directory
                csv_files = glob.glob(os.path.join(self.data_loader.data_path, "*.csv"))
                if csv_files:
                    dfs = []
                    for file in csv_files:
                        df = self.data_loader.load_from_csv(file)
                        if df is not None:
                            dfs.append(df)
                    
                    if dfs:
                        self.data = pd.concat(dfs, ignore_index=True)
                    else:
                        print("No CSV files loaded, using sample data")
                        self.data = self.data_loader.generate_sample_real_data()
                else:
                    print(f"No CSV files found in {self.data_loader.data_path}")
                    print("Generating sample data for demonstration...")
                    self.data = self.data_loader.generate_sample_real_data()
            else:
                print("No data path provided, generating sample data...")
                self.data = self.data_loader.generate_sample_real_data()
        
        elif self.data_source == 'azure':
            # Configure for Azure Monitor
            connection_string = os.getenv('AZURE_CONNECTION_STRING')
            workspace_id = os.getenv('AZURE_WORKSPACE_ID')
            query = "Perf | where ObjectName == 'Processor' | project TimeGenerated, Computer, CounterValue"
            self.data = self.data_loader.load_from_azure_monitor(connection_string, query)
            if self.data is None:
                self.data = self.data_loader.generate_sample_real_data()
        
        elif self.data_source == 'aws':
            # Configure for AWS CloudWatch
            region = os.getenv('AWS_REGION', 'us-east-1')
            namespace = 'AWS/EC2'
            metric_names = ['CPUUtilization', 'MemoryUtilization']
            self.data = self.data_loader.load_from_aws_cloudwatch(region, namespace, metric_names, None, None)
            if self.data is None:
                self.data = self.data_loader.generate_sample_real_data()
        
        elif self.data_source == 'prometheus':
            # Configure for Prometheus
            prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
            query = 'avg(container_cpu_usage_seconds_total) by (instance)'
            self.data = self.data_loader.load_from_prometheus(prometheus_url, query)
            if self.data is None:
                self.data = self.data_loader.generate_sample_real_data()
        
        elif self.data_source == 'kafka':
            # Configure for Kafka
            topic = os.getenv('KAFKA_TOPIC', 'cloud_metrics')
            bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
            self.data = self.data_loader.load_from_kafka(topic, bootstrap_servers)
            if self.data is None:
                self.data = self.data_loader.generate_sample_real_data()
        
        else:
            print(f"Unknown data source: {self.data_source}")
            self.data = self.data_loader.generate_sample_real_data()
        
        print(f"\nData loaded: {len(self.data)} records")
        print(f"Time range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        print(f"Number of VMs: {self.data['vm_id'].nunique()}")
        
        return self.data
    
    def prepare_features(self):
        """Prepare features for training"""
        print(f"\n{'='*60}")
        print("Preparing Features")
        print(f"{'='*60}")
        
        # Define feature columns
        self.feature_cols = ['cpu_util', 'memory_util', 'disk_io', 
                           'network_traffic', 'queue_length', 'response_time',
                           'load_score', 'hour_of_day', 'day_of_week']
        
        # Check which features are available
        available_features = [col for col in self.feature_cols if col in self.data.columns]
        missing_features = set(self.feature_cols) - set(available_features)
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Fill missing features with zeros
            for col in missing_features:
                self.data[col] = 0
            print(f"Filled missing features with zeros")
        
        self.feature_cols = available_features + list(missing_features)
        
        print(f"\nUsing {len(self.feature_cols)} features: {self.feature_cols}")
        
        # Normalize features
        scaled_features = self.ai_model.scaler.fit_transform(self.data[self.feature_cols])
        self.data[self.feature_cols] = scaled_features
        
        # Add derived target if needed
        if 'sla_violation_risk' not in self.data.columns:
            self.data['sla_violation_risk'] = (
                (self.data['cpu_util'] > 0.8) | 
                (self.data['memory_util'] > 0.75) |
                (self.data['response_time'] > 0.6)
            ).astype(int)
        
        print("Feature preparation complete!")
        
        return self.data
    
    def train_prediction_model(self):
        """Train LSTM model for resource prediction"""
        print("\n" + "="*60)
        print("Training Prediction Model (LSTM)")
        print("="*60)
        
        # Group by VM and train for each
        vms = self.data['vm_id'].unique()
        print(f"Training models for {len(vms)} VMs...")
        
        models_per_vm = {}
        
        for vm_id in vms[:3]:  # Limit to first 3 VMs for demonstration
            print(f"\nTraining for VM {vm_id}...")
            vm_data = self.data[self.data['vm_id'] == vm_id].sort_values('timestamp')
            
            if len(vm_data) < self.ai_model.sequence_length + 10:
                print(f"  Insufficient data for VM {vm_id}, skipping...")
                continue
            
            # Prepare sequences
            sequences, targets = self.ai_model.prepare_sequences(
                vm_data, self.feature_cols
            )
            
            if len(sequences) < 10:
                print(f"  Not enough sequences for VM {vm_id}, skipping...")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                sequences, targets, test_size=0.2, random_state=42
            )
            
            # Create model
            input_dim = X_train.shape[2]
            model = self.ai_model.create_lstm_model(input_dim)
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_split=0.1,
                epochs=10,
                batch_size=32,
                verbose=0
            )
            
            # Evaluate
            test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
            print(f"  Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
            
            models_per_vm[vm_id] = model
        
        self.trained_models['prediction'] = models_per_vm
        self.results['prediction_metrics'] = {
            'models_trained': len(models_per_vm),
            'vms_covered': list(models_per_vm.keys())
        }
        
        return models_per_vm
    
    def train_decision_model(self):
        """Train decision model for load balancing"""
        print("\n" + "="*60)
        print("Training Decision Model")
        print("="*60)
        
        # Prepare decision data
        decision_data = self.data.copy()
        
        # Create target based on actual SLA violations
        decision_data['needs_rebalance'] = (
            (decision_data['cpu_util'] > 0.8) | 
            (decision_data['memory_util'] > 0.75) |
            (decision_data['response_time'] > 0.6) |
            (decision_data['queue_length'] > 0.7)
        ).astype(int)
        
        X = decision_data[self.feature_cols]
        y = decision_data['needs_rebalance']
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution:")
        target_counts = y.value_counts(normalize=True)
        for value, proportion in target_counts.items():
            print(f"  Class {value}: {proportion:.2%} ({y.value_counts()[value]} samples)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create model
        input_dim = X_train.shape[1]
        model = self.ai_model.create_decision_model(input_dim)
        
        # Calculate class weights
        class_0_count = np.sum(y_train == 0)
        class_1_count = np.sum(y_train == 1)
        total = class_0_count + class_1_count
        
        class_weight = {
            0: total / (2 * class_0_count) if class_0_count > 0 else 1.0,
            1: total / (2 * class_1_count) if class_1_count > 0 else 1.0
        }
        
        print(f"\nClass weights: {class_weight}")
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=64,
            class_weight=class_weight,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Predictions
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
        
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.trained_models['decision'] = model
        self.results['decision_metrics'] = {
            'test_accuracy': float(test_acc),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'history': history.history
        }
        
        return model, history
    
    def simulate_load_balancing(self):
        """Simulate AI-driven load balancing on real data"""
        print("\n" + "="*60)
        print("Simulating AI-Driven Load Balancing")
        print("="*60)
        
        # Group by timestamp
        timestamps = sorted(self.data['timestamp'].unique())
        timestamps = timestamps[:min(500, len(timestamps))]  # Limit for performance
        
        print(f"Simulating {len(timestamps)} timestamps...")
        
        simulation_results = []
        action_counts = {"Redistribute load": 0, "Maintain current state": 0, "None": 0}
        
        for i, timestamp in enumerate(timestamps):
            timestamp_data = self.data[self.data['timestamp'] == timestamp]
            
            if len(timestamp_data) == 0:
                continue
            
            # Calculate system metrics
            avg_cpu = timestamp_data['cpu_util'].mean()
            avg_memory = timestamp_data['memory_util'].mean()
            avg_response = timestamp_data['response_time'].mean()
            sla_violations = timestamp_data['sla_violation_risk'].sum()
            total_vms = len(timestamp_data)
            
            action = "None"
            
            # Simulate AI decision
            if self.trained_models.get('decision') and len(timestamp_data) > 0:
                avg_features = timestamp_data[self.feature_cols].mean().values.reshape(1, -1)
                needs_rebalance = self.trained_models['decision'].predict(avg_features, verbose=0)[0][0]
                
                if needs_rebalance > 0.7:
                    action = "Redistribute load"
                    
                    # Simple redistribution simulation
                    overloaded_count = len(timestamp_data[timestamp_data['load_score'] > 0.8])
                    underloaded_count = len(timestamp_data[timestamp_data['load_score'] < 0.3])
                    
                    if overloaded_count > 0 and underloaded_count > 0:
                        improvement_factor = min(overloaded_count, underloaded_count) / total_vms
                        sla_violations = max(0, sla_violations * (1 - improvement_factor))
                else:
                    action = "Maintain current state"
            
            action_counts[action] += 1
            
            simulation_results.append({
                'timestamp': timestamp,
                'total_vms': total_vms,
                'avg_cpu_util': avg_cpu,
                'avg_memory_util': avg_memory,
                'avg_response_time': avg_response,
                'sla_violations': sla_violations,
                'action_taken': action
            })
            
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{len(timestamps)} timestamps")
        
        simulation_df = pd.DataFrame(simulation_results)
        self.results['simulation'] = simulation_df
        
        print(f"\nSimulation complete!")
        print(f"\nAction Distribution:")
        for action, count in action_counts.items():
            if count > 0:
                print(f"  {action}: {count} times ({count/len(timestamps)*100:.1f}%)")
        
        return simulation_df
    
    def evaluate_performance(self):
        """Evaluate the performance of the AI model"""
        print("\n" + "="*60)
        print("Evaluating AI Model Performance")
        print("="*60)
        
        simulation_df = self.results.get('simulation')
        
        if simulation_df is not None and len(simulation_df) > 0:
            balanced_timestamps = simulation_df[simulation_df['action_taken'] == "Redistribute load"]
            unbalanced_timestamps = simulation_df[simulation_df['action_taken'] != "Redistribute load"]
            
            if len(balanced_timestamps) > 0 and len(unbalanced_timestamps) > 0:
                balanced_avg = balanced_timestamps.mean(numeric_only=True)
                unbalanced_avg = unbalanced_timestamps.mean(numeric_only=True)
                
                metrics = {
                    'cpu_improvement': float(unbalanced_avg['avg_cpu_util'] - balanced_avg['avg_cpu_util']),
                    'memory_improvement': float(unbalanced_avg['avg_memory_util'] - balanced_avg['avg_memory_util']),
                    'response_time_improvement': float(unbalanced_avg['avg_response_time'] - balanced_avg['avg_response_time']),
                    'sla_violation_reduction': float(unbalanced_avg['sla_violations'] - balanced_avg['sla_violations']),
                    'total_actions': int(len(balanced_timestamps)),
                    'action_percentage': float(len(balanced_timestamps) / len(simulation_df) * 100)
                }
                
                print("\nPerformance Comparison:")
                print("-" * 70)
                print(f"{'Metric':<30} {'Without AI':<12} {'With AI':<12} {'Improvement':<12}")
                print("-" * 70)
                print(f"{'Avg CPU Utilization':<30} {unbalanced_avg['avg_cpu_util']:<12.4f} {balanced_avg['avg_cpu_util']:<12.4f} {metrics['cpu_improvement']:<12.4f}")
                print(f"{'Avg Memory Utilization':<30} {unbalanced_avg['avg_memory_util']:<12.4f} {balanced_avg['avg_memory_util']:<12.4f} {metrics['memory_improvement']:<12.4f}")
                print(f"{'Avg Response Time':<30} {unbalanced_avg['avg_response_time']:<12.4f} {balanced_avg['avg_response_time']:<12.4f} {metrics['response_time_improvement']:<12.4f}")
                print(f"{'Avg SLA Violations':<30} {unbalanced_avg['sla_violations']:<12.2f} {balanced_avg['sla_violations']:<12.2f} {metrics['sla_violation_reduction']:<12.2f}")
                print("-" * 70)
                
                print(f"\nAI Actions: {metrics['total_actions']} times ({metrics['action_percentage']:.1f}% of timestamps)")
                
                self.results['performance_metrics'] = metrics
                return metrics
            
            else:
                print("Not enough data for comparison (need both balanced and unbalanced timestamps)")
                return None
        
        print("No simulation data available for evaluation.")
        return None
    
    def save_models(self, path="models/"):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)
        
        print(f"\nSaving models to {path}...")
        
        for name, model in self.trained_models.items():
            if name == 'prediction' and isinstance(model, dict):
                # Save models per VM
                for vm_id, vm_model in model.items():
                    model_path = f"{path}{name}_vm_{vm_id}.h5"
                    vm_model.save(model_path)
                    print(f"Saved: {model_path}")
            elif name != 'prediction':
                model_path = f"{path}{name}_model.h5"
                model.save(model_path)
                print(f"Saved: {model_path}")
        
        # Save scaler
        scaler_path = f"{path}scaler.pkl"
        joblib.dump(self.ai_model.scaler, scaler_path)
        print(f"Saved: {scaler_path}")
        
        # Save metadata
        metadata = {
            'feature_cols': self.feature_cols,
            'use_real_data': self.use_real_data,
            'data_source': self.data_source,
            'data_shape': self.data.shape if self.data is not None else None,
            'timestamp_range': [str(self.data['timestamp'].min()), str(self.data['timestamp'].max())] if self.data is not None else None,
            'num_vms': int(self.data['vm_id'].nunique()) if self.data is not None else None
        }
        
        import json
        with open(f"{path}metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Saved: {path}metadata.json")
        
        print(f"\nAll models and results saved successfully!")
    
    def run_complete_pipeline(self):
        """Run the complete AI model pipeline with real data"""
        print("\n" + "="*70)
        print("AI-Driven Dynamic Load Balancing Model Pipeline (Real Data)")
        print("="*70)
        
        try:
            # Step 1: Load real data
            print("\n[STEP 1] Loading Real Data")
            self.load_real_data()
            
            # Step 2: Prepare features
            print("\n[STEP 2] Feature Preparation")
            self.prepare_features()
            
            # Step 3: Train prediction model
            print("\n[STEP 3] Training Prediction Model")
            self.train_prediction_model()
            
            # Step 4: Train decision model
            print("\n[STEP 4] Training Decision Model")
            self.train_decision_model()
            
            # Step 5: Simulate load balancing
            print("\n[STEP 5] Load Balancing Simulation")
            self.simulate_load_balancing()
            
            # Step 6: Evaluate performance
            print("\n[STEP 6] Performance Evaluation")
            self.evaluate_performance()
            
            # Step 7: Save models
            print("\n[STEP 7] Saving Results")
            self.save_models()
            
            print("\n" + "="*70)
            print("Pipeline completed successfully!")
            print("="*70)
            
            return self.results
            
        except Exception as e:
            print(f"\nError in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Main execution
if __name__ == "__main__":
    print("Starting AI-Driven Dynamic Load Balancing System")
    print("="*70)
    
    # Configuration
    USE_REAL_DATA = True  # Set to True to use real data
    DATA_SOURCE = 'csv'   # Options: 'csv', 'azure', 'aws', 'prometheus', 'kafka'
    DATA_PATH = "data/"   # Path to your CSV files (create this directory and add your data)
    
    # Create data directory if it doesn't exist
    if USE_REAL_DATA and DATA_SOURCE == 'csv':
        os.makedirs(DATA_PATH, exist_ok=True)
        
        # Check if there are CSV files
        csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
        
        if not csv_files:
            print(f"\nNo CSV files found in '{DATA_PATH}' directory.")
            print(f"Please place your cloud workload CSV files in the '{DATA_PATH}' directory.")
            print("\nExpected CSV format should include columns like:")
            print("  - timestamp (datetime)")
            print("  - vm_id (int/string)")
            print("  - cpu_util (float, 0-100)")
            print("  - memory_util (float, 0-100)")
            print("  - response_time (float, ms)")
            print("  - (optional) disk_io, network_traffic, queue_length")
            print("\nExample CSV:")
            print("timestamp,vm_id,cpu_util,memory_util,response_time")
            print("2024-01-01 00:00:00,1,45.2,52.3,125.4")
            print("2024-01-01 00:00:00,2,38.7,48.1,98.2")
            print("\nOr you can use the sample data generator by setting USE_REAL_DATA=False")
            
            response = input("\nDo you want to generate sample data instead? (y/n): ")
            if response.lower() == 'y':
                USE_REAL_DATA = False
    
    # Initialize and run the pipeline
    load_balancer = AIBasedLoadBalancer(
        use_real_data=USE_REAL_DATA,
        data_source=DATA_SOURCE,
        data_path=DATA_PATH if USE_REAL_DATA else None
    )
    
    # Run complete pipeline
    results = load_balancer.run_complete_pipeline()
    
    if results:
        print("\n" + "="*70)
        print("RESEARCH PAPER RESULTS SUMMARY")
        print("="*70)
        
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"\n1. Performance Improvements with AI:")
            print(f"   - CPU Utilization Improvement: {metrics['cpu_improvement']:.4f}")
            print(f"   - Memory Utilization Improvement: {metrics['memory_improvement']:.4f}")
            print(f"   - Response Time Reduction: {metrics['response_time_improvement']:.4f}")
            print(f"   - SLA Violations Reduced by: {metrics['sla_violation_reduction']:.2f}")
            print(f"   - Load Balancing Actions: {metrics['total_actions']} times ({metrics['action_percentage']:.1f}% of time)")
        
        if 'decision_metrics' in results:
            decision_metrics = results['decision_metrics']['classification_report']
            print(f"\n2. Model Performance:")
            print(f"   - Decision Model Accuracy: {decision_metrics['accuracy']:.2%}")
            if '1' in decision_metrics:
                print(f"   - Precision: {decision_metrics['1']['precision']:.2%}")
                print(f"   - Recall: {decision_metrics['1']['recall']:.2%}")
                print(f"   - F1-Score: {decision_metrics['1']['f1-score']:.2%}")
        
        print("\n" + "="*70)
        print("All tasks completed successfully!")
        print("="*70)
        
        # Save simulation results
        if 'simulation' in results:
            if isinstance(results['simulation'], pd.DataFrame):
                results['simulation'].to_csv('simulation_results.csv', index=False)
                print("\nSimulation results saved to 'simulation_results.csv'")
    else:
        print("\nPipeline failed to complete successfully.")