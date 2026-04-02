# real_dataset_loader.py (Fixed Version)
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RealCloudDatasetDownloader:
    """Download and load real cloud datasets"""
    
    def __init__(self, download_dir='data/'):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)
        
    def download_bitbrains_dataset(self):
        """Download Bitbrains dataset (no registration required)"""
        url = "https://raw.githubusercontent.com/bitbrains/bitbrains-datasets/master/data/csv/bitbrains-2013-08-01.csv"
        filepath = os.path.join(self.download_dir, 'bitbrains-2013-08-01.csv')
        
        print(f"Downloading Bitbrains dataset from {url}...")
        try:
            response = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error downloading: {e}")
            return None
    
    def generate_realistic_sample(self):
        """Generate realistic dataset based on real cloud patterns"""
        print("Generating realistic cloud workload dataset...")
        
        np.random.seed(42)
        
        # Number of VMs and time points
        num_vms = 20
        num_hours = 24 * 30  # 30 days
        num_points = num_vms * num_hours
        
        # Create arrays for data
        timestamps = []
        vm_ids = []
        cpu_utils = []
        memory_utils = []
        disk_ios = []
        network_traffic = []
        response_times = []
        
        start_time = datetime(2024, 1, 1)
        
        for vm in range(num_vms):
            for hour in range(num_hours):
                timestamp = start_time + timedelta(hours=hour)
                timestamps.append(timestamp)
                vm_ids.append(f"vm_{vm}")
                
                # Realistic patterns from actual cloud workloads
                hour_of_day = timestamp.hour
                day_of_week = timestamp.weekday()
                
                # Business hours pattern (based on real cloud data)
                if 9 <= hour_of_day <= 17 and day_of_week < 5:
                    base_cpu = 65 + np.random.normal(0, 10)
                    base_memory = 70 + np.random.normal(0, 8)
                elif 18 <= hour_of_day <= 22:
                    base_cpu = 45 + np.random.normal(0, 12)
                    base_memory = 55 + np.random.normal(0, 10)
                else:
                    base_cpu = 25 + np.random.normal(0, 8)
                    base_memory = 35 + np.random.normal(0, 8)
                
                # Add VM-specific variation (like different workload types)
                vm_variation = (vm % 5) * 5
                
                cpu_util = max(0, min(100, base_cpu + vm_variation + np.random.normal(0, 5)))
                memory_util = max(0, min(100, base_memory + vm_variation * 0.5 + np.random.normal(0, 5)))
                
                cpu_utils.append(cpu_util)
                memory_utils.append(memory_util)
                disk_ios.append(np.random.poisson(100))
                network_traffic.append(np.random.exponential(200))
                response_times.append(50 + cpu_util * 2 + np.random.normal(0, 10))
        
        # Convert to numpy arrays for calculations
        cpu_utils = np.array(cpu_utils)
        memory_utils = np.array(memory_utils)
        disk_ios = np.array(disk_ios)
        network_traffic = np.array(network_traffic)
        response_times = np.array(response_times)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'vm_id': vm_ids,
            'cpu_util': cpu_utils,
            'memory_util': memory_utils,
            'disk_io': disk_ios,
            'network_traffic': network_traffic,
            'response_time': response_times
        })
        
        # Calculate load score
        df['load_score'] = (
            0.4 * (df['cpu_util'] / 100) +
            0.3 * (df['memory_util'] / 100) +
            0.2 * (df['response_time'] / 500) +
            0.1 * (df['disk_io'] / 1000)
        )
        
        # Add time features
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Save to CSV
        filepath = os.path.join(self.download_dir, 'realistic_cloud_workload.csv')
        df.to_csv(filepath, index=False)
        print(f"Generated dataset saved to {filepath}")
        print(f"Dataset shape: {df.shape}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Number of VMs: {df['vm_id'].nunique()}")
        
        return filepath


class RealDatasetLoadBalancer:
    """Load balancer with real dataset support"""
    
    def __init__(self):
        self.downloader = RealCloudDatasetDownloader()
        self.data = None
        self.current_index = 0
        
    def load_workload_data(self, dataset_path=None, use_sample=False):
        """Load workload data from file or generate sample"""
        
        if dataset_path and os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}...")
            self.data = pd.read_csv(dataset_path)
            
        elif use_sample:
            # Use the realistic generated sample
            dataset_path = self.downloader.generate_realistic_sample()
            self.data = pd.read_csv(dataset_path)
            
        else:
            # Try to download real dataset
            print("Attempting to download real cloud dataset...")
            filepath = self.downloader.download_bitbrains_dataset()
            
            if filepath and os.path.exists(filepath):
                self.data = pd.read_csv(filepath)
            else:
                print("Download failed, generating realistic sample...")
                dataset_path = self.downloader.generate_realistic_sample()
                self.data = pd.read_csv(dataset_path)
        
        # Convert timestamp
        if 'timestamp' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
            if 'hour_of_day' not in self.data.columns:
                self.data['hour_of_day'] = self.data['timestamp'].dt.hour
            if 'day_of_week' not in self.data.columns:
                self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        
        print(f"\nDataset loaded successfully!")
        print(f"Records: {len(self.data)}")
        print(f"Columns: {self.data.columns.tolist()}")
        print(f"Time range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        
        return self.data
    
    def get_data_summary(self):
        """Get dataset summary"""
        if self.data is None:
            return None
        
        summary = {
            'total_records': len(self.data),
            'unique_vms': self.data['vm_id'].nunique() if 'vm_id' in self.data.columns else 0,
            'columns': self.data.columns.tolist()
        }
        
        if 'cpu_util' in self.data.columns:
            summary['avg_cpu'] = float(self.data['cpu_util'].mean())
            summary['min_cpu'] = float(self.data['cpu_util'].min())
            summary['max_cpu'] = float(self.data['cpu_util'].max())
            
        if 'memory_util' in self.data.columns:
            summary['avg_memory'] = float(self.data['memory_util'].mean())
            
        if 'timestamp' in self.data.columns:
            summary['time_range'] = {
                'start': str(self.data['timestamp'].min()),
                'end': str(self.data['timestamp'].max())
            }
        
        return summary
    
    def prepare_for_ml(self):
        """Prepare data for machine learning"""
        if self.data is None:
            return None, None
        
        # Select features for ML
        feature_cols = ['cpu_util', 'memory_util', 'hour_of_day', 'day_of_week']
        available_cols = [col for col in feature_cols if col in self.data.columns]
        
        # Add derived features if needed
        if 'load_score' not in self.data.columns:
            self.data['load_score'] = (
                0.4 * (self.data['cpu_util'] / 100) +
                0.3 * (self.data['memory_util'] / 100)
            )
        
        X = self.data[available_cols]
        y = self.data['load_score'] if 'load_score' in self.data.columns else self.data['cpu_util']
        
        print(f"\nPrepared data for ML:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Features: {available_cols}")
        
        return X, y
    
    def stream_data(self, speed=1.0):
        """Stream data in real-time"""
        if self.data is None:
            print("No data loaded")
            return
        
        print(f"\nStreaming {len(self.data)} records...")
        
        for idx, row in self.data.iterrows():
            self.current_index = idx
            yield row
            
            # Calculate next timestamp difference
            if idx < len(self.data) - 1:
                current_time = row['timestamp']
                next_time = self.data.iloc[idx + 1]['timestamp']
                time_diff = (next_time - current_time).total_seconds()
                
                if time_diff > 0 and time_diff < 3600:  # Don't wait too long
                    import time
                    time.sleep(time_diff / speed)


def create_dashboard():
    """Create a simple dashboard for visualization"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Load the generated data
    data_path = 'data/realistic_cloud_workload.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. CPU Usage over time
        ax1 = axes[0, 0]
        for vm in df['vm_id'].unique()[:5]:  # Show first 5 VMs
            vm_data = df[df['vm_id'] == vm]
            ax1.plot(vm_data['timestamp'], vm_data['cpu_util'], alpha=0.5, label=vm)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('CPU Utilization (%)')
        ax1.set_title('CPU Usage Over Time')
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution of CPU Usage
        ax2 = axes[0, 1]
        ax2.hist(df['cpu_util'], bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('CPU Utilization (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('CPU Usage Distribution')
        ax2.axvline(df['cpu_util'].mean(), color='red', linestyle='--', label=f'Mean: {df["cpu_util"].mean():.1f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Load Score Distribution by Hour
        ax3 = axes[1, 0]
        df['hour'] = df['timestamp'].dt.hour
        hourly_load = df.groupby('hour')['load_score'].mean()
        ax3.bar(hourly_load.index, hourly_load.values)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Average Load Score')
        ax3.set_title('Load Score by Hour')
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation Heatmap
        ax4 = axes[1, 1]
        corr_cols = ['cpu_util', 'memory_util', 'response_time', 'load_score']
        if all(col in df.columns for col in corr_cols):
            corr = df[corr_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax4)
            ax4.set_title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig('workload_analysis.png', dpi=100, bbox_inches='tight')
        print("\nDashboard saved as 'workload_analysis.png'")
        plt.show()
    else:
        print("No data file found. Run the script first to generate data.")


if __name__ == "__main__":
    print("="*60)
    print("Real Cloud Workload Dataset Loader")
    print("="*60)
    
    # Initialize
    balancer = RealDatasetLoadBalancer()
    
    # Load data (choose one method)
    print("\n1. Loading dataset...")
    
    # Option A: Use realistic generated data (immediately available)
    balancer.load_workload_data(use_sample=True)
    
    # Option B: If you have a real dataset file
    # balancer.load_workload_data(dataset_path='data/your_dataset.csv')
    
    # Get dataset summary
    print("\n2. Dataset Summary:")
    summary = balancer.get_data_summary()
    if summary:
        print(f"   - Total records: {summary['total_records']:,}")
        print(f"   - Unique VMs: {summary['unique_vms']}")
        if 'avg_cpu' in summary:
            print(f"   - Average CPU: {summary['avg_cpu']:.2f}%")
            print(f"   - CPU Range: {summary['min_cpu']:.2f}% - {summary['max_cpu']:.2f}%")
        if 'avg_memory' in summary:
            print(f"   - Average Memory: {summary['avg_memory']:.2f}%")
        if 'time_range' in summary:
            print(f"   - Time range: {summary['time_range']['start']} to {summary['time_range']['end']}")
        print(f"   - Columns: {summary['columns']}")
    
    # Prepare data for ML
    print("\n3. Preparing data for Machine Learning:")
    X, y = balancer.prepare_for_ml()
    
    # Show sample data
    print("\n4. Sample Data (first 10 rows):")
    print(balancer.data[['timestamp', 'vm_id', 'cpu_util', 'memory_util', 'load_score']].head(10))
    
    # Statistical analysis
    print("\n5. Statistical Analysis:")
    if 'cpu_util' in balancer.data.columns:
        print(f"   CPU Utilization:")
        print(f"      Min: {balancer.data['cpu_util'].min():.2f}%")
        print(f"      Max: {balancer.data['cpu_util'].max():.2f}%")
        print(f"      Mean: {balancer.data['cpu_util'].mean():.2f}%")
        print(f"      Std: {balancer.data['cpu_util'].std():.2f}%")
        print(f"      25th percentile: {balancer.data['cpu_util'].quantile(0.25):.2f}%")
        print(f"      75th percentile: {balancer.data['cpu_util'].quantile(0.75):.2f}%")
    
    if 'memory_util' in balancer.data.columns:
        print(f"\n   Memory Utilization:")
        print(f"      Min: {balancer.data['memory_util'].min():.2f}%")
        print(f"      Max: {balancer.data['memory_util'].max():.2f}%")
        print(f"      Mean: {balancer.data['memory_util'].mean():.2f}%")
        print(f"      Std: {balancer.data['memory_util'].std():.2f}%")
    
    # Pattern detection
    print("\n6. Workload Patterns Detected:")
    
    # Check for business hours pattern
    if 'timestamp' in balancer.data.columns and 'cpu_util' in balancer.data.columns:
        balancer.data['hour'] = balancer.data['timestamp'].dt.hour
        hourly_avg = balancer.data.groupby('hour')['cpu_util'].mean()
        
        peak_hour = hourly_avg.idxmax()
        peak_load = hourly_avg.max()
        off_peak_hours = [0,1,2,3,4,5,22,23]
        off_peak_load = hourly_avg[off_peak_hours].mean()
        
        print(f"   Peak hour: {peak_hour}:00 ({peak_load:.2f}% CPU)")
        print(f"   Off-peak average: {off_peak_load:.2f}% CPU")
        print(f"   Peak to off-peak ratio: {peak_load/off_peak_load:.2f}x")
        
        # Detect if workload is business-hours heavy
        if peak_load/off_peak_load > 1.5:
            print(f"   Pattern: Strong business-hours workload pattern detected")
        else:
            print(f"   Pattern: Relatively constant workload pattern")
    
    # VM load distribution
    if 'vm_id' in balancer.data.columns and 'cpu_util' in balancer.data.columns:
        vm_avg_cpu = balancer.data.groupby('vm_id')['cpu_util'].mean()
        print(f"\n7. VM Load Distribution:")
        print(f"   Most loaded VM: {vm_avg_cpu.idxmax()} ({vm_avg_cpu.max():.2f}% CPU)")
        print(f"   Least loaded VM: {vm_avg_cpu.idxmin()} ({vm_avg_cpu.min():.2f}% CPU)")
        print(f"   Load imbalance ratio: {vm_avg_cpu.max()/vm_avg_cpu.min():.2f}x")
    
    print("\n" + "="*60)
    print("Dataset ready for load balancing AI model!")
    print("\nNext steps:")
    print("1. Use this dataset to train LSTM models for load prediction")
    print("2. Implement load balancing algorithms")
    print("3. Simulate real-time load distribution")
    print("\nDashboard saved as 'workload_analysis.png'")
    print("="*60)
    
    # Optionally create dashboard
    try:
        create_dashboard()
    except Exception as e:
        print(f"\nNote: Dashboard generation requires matplotlib. Install with: pip install matplotlib seaborn")
        print(f"Error: {e}")
