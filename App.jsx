import React, { useState, useEffect, useRef } from 'react';
import { Play, Settings, AlertCircle, CheckCircle, Clock, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:5001/api';

export default function DatasetConverterUI() {
  const [sourceDataset, setSourceDataset] = useState('idd3d');
  const [targetDataset, setTargetDataset] = useState('nuscenes');
  const [rootPath, setRootPath] = useState('/home/siddharthb9/Desktop/nuSceneses&IDD3D');
  const [sequenceId, setSequenceId] = useState('20220118103308_seq_10');
  const [conversions, setConversions] = useState({
    lidar: true,
    calib: true,
    annot: true,
    maps: false,
    egoPose: false
  });
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [progress, setProgress] = useState(0);
  const [pathValidation, setPathValidation] = useState(null);
  const [isValidating, setIsValidating] = useState(false);
  const logsEndRef = useRef(null);

  const datasets = {
    idd3d: {
      name: 'IDD3D (Indian Dataset)',
      description: '6 cameras, 1 LiDAR, GPS - 10Hz camera, 10Hz LiDAR',
      sensors: '6 RGB cameras, 64-channel LiDAR, GPS',
      format: 'PCD (lidar), PNG (camera), JSON (annotations)'
    },
    nuscenes: {
      name: 'nuScenes',
      description: '6 cameras, 1 LiDAR, 5 RADARs, GPS/IMU - 12Hz camera, 20Hz LiDAR',
      sensors: '6 cameras, Velodyne HDL-32E, 5 RADARs, GPS/IMU',
      format: 'JPEG (camera), .pcd.bin (lidar), JSON (metadata)'
    }
  };

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { message, type, timestamp }]);
  };

  const handleValidatePaths = async () => {
    setIsValidating(true);
    setPathValidation(null);
    try {
      const response = await fetch(`${API_BASE}/validate-paths`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ root_path: rootPath, sequence_id: sequenceId })
      });
      const data = await response.json();
      setPathValidation(data);
    } catch (error) {
      setPathValidation({ valid: false, error: error.message });
    } finally {
      setIsValidating(false);
    }
  };

  const handleRunConversion = async () => {
    if (!rootPath.trim()) {
      addLog('Error: Root path is required', 'error');
      return;
    }

    setIsRunning(true);
    setLogs([]);
    setProgress(0);
    addLog(`Starting conversion: ${sourceDataset} → ${targetDataset}`, 'info');
    addLog(`Root path: ${rootPath}`, 'info');
    addLog(`Sequence ID: ${sequenceId}`, 'info');

    const activeConversions = Object.entries(conversions)
      .filter(([_, enabled]) => enabled)
      .map(([name, _]) => name);

    if (activeConversions.length === 0) {
      addLog('Error: No conversion modules selected', 'error');
      setIsRunning(false);
      return;
    }

    addLog(`Active conversions: ${activeConversions.join(', ')}`, 'info');

    try {
      const response = await fetch(`${API_BASE}/convert/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          root_path: rootPath,
          sequence_id: sequenceId,
          conversions: conversions
        })
      });

      if (!response.ok) {
        const error = await response.json();
        addLog(`Error: ${error.error}`, 'error');
        setIsRunning(false);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines[lines.length - 1];

        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i];
          if (line.startsWith('data: ')) {
            try {
              const logEntry = JSON.parse(line.slice(6));
              if (logEntry.type === 'complete') {
                setProgress(100);
              } else {
                addLog(logEntry.message, logEntry.type);
              }
            } catch (e) {
              // Skip parse errors
            }
          }
        }
      }

      addLog('✓ Conversion pipeline completed successfully!', 'success');
      addLog(`Output directory: ${rootPath}/Intermediate_format/`, 'info');
    } catch (error) {
      addLog(`Error: ${error.message}`, 'error');
    } finally {
      setIsRunning(false);
    }
  };

  const getLogColor = (type) => {
    switch(type) {
      case 'success': return 'text-green-600';
      case 'error': return 'text-red-600';
      case 'warning': return 'text-yellow-600';
      case 'info': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Dataset Converter</h1>
          <p className="text-gray-600">Convert between autonomous driving dataset formats</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Settings size={20} /> Source Dataset
              </h2>
              
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Select Dataset</label>
                  <select 
                    value={sourceDataset}
                    onChange={(e) => setSourceDataset(e.target.value)}
                    disabled={isRunning}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                  >
                    <option value="idd3d">IDD3D</option>
                  </select>
                </div>

                <div className="bg-blue-50 rounded p-3 text-sm">
                  <p className="font-semibold text-blue-900 mb-1">{datasets[sourceDataset].name}</p>
                  <p className="text-blue-800 text-xs mb-2">{datasets[sourceDataset].description}</p>
                  <p className="text-blue-700 text-xs"><strong>Sensors:</strong> {datasets[sourceDataset].sensors}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Target Format</h2>
              
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Select Format</label>
                  <select 
                    value={targetDataset}
                    onChange={(e) => setTargetDataset(e.target.value)}
                    disabled={isRunning}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                  >
                    <option value="nuscenes">nuScenes</option>
                  </select>
                </div>

                <div className="bg-green-50 rounded p-3 text-sm">
                  <p className="font-semibold text-green-900 mb-1">{datasets[targetDataset].name}</p>
                  <p className="text-green-800 text-xs mb-2">{datasets[targetDataset].description}</p>
                  <p className="text-green-700 text-xs"><strong>Format:</strong> {datasets[targetDataset].format}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
                <Settings size={20} />
                Paths
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Root Directory</label>
                  <input 
                    type="text"
                    value={rootPath}
                    onChange={(e) => setRootPath(e.target.value)}
                    disabled={isRunning}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Sequence ID</label>
                  <input 
                    type="text"
                    value={sequenceId}
                    onChange={(e) => setSequenceId(e.target.value)}
                    disabled={isRunning}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
                  />
                </div>

                <button
                  onClick={handleValidatePaths}
                  disabled={isRunning || isValidating}
                  className="w-full bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white font-semibold py-2 rounded-lg transition flex items-center justify-center gap-2 text-sm"
                >
                  <RefreshCw size={16} />
                  {isValidating ? 'Validating...' : 'Validate Paths'}
                </button>

                {pathValidation && (
                  <div className={`p-3 rounded text-sm ${
                    pathValidation.valid 
                      ? 'bg-green-50 border border-green-200' 
                      : 'bg-red-50 border border-red-200'
                  }`}>
                    {pathValidation.valid ? (
                      <div className="text-green-800 space-y-1">
                        <p className="font-semibold">✓ Paths valid</p>
                        <p className="text-xs">LiDAR files: {pathValidation.lidar_files}</p>
                        <p className="text-xs">Label files: {pathValidation.label_files}</p>
                      </div>
                    ) : (
                      <div className="text-red-800">
                        <p className="font-semibold">✗ Validation failed</p>
                        <p className="text-xs">{pathValidation.error}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Conversion Modules</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {[
                  { key: 'lidar', label: 'LiDAR Conversion', desc: 'Convert .pcd to .pcd.bin' },
                  { key: 'calib', label: 'Calibration Stubs', desc: 'Generate sensor calibration' },
                  { key: 'annot', label: 'Annotations', desc: 'Convert frame annotations' },
                  { key: 'maps', label: 'Maps (Disabled)', desc: 'Generate HD maps - unavailable' },
                  { key: 'egoPose', label: 'Ego Pose (Disabled)', desc: 'Generate ego trajectory - unavailable' }
                ].map(({ key, label, desc }) => {
                  const isDisabled = ['maps', 'egoPose'].includes(key);
                  return (
                    <label key={key} className={`flex items-start gap-3 p-3 border rounded-lg cursor-pointer transition ${
                      isDisabled 
                        ? 'bg-gray-50 border-gray-200 opacity-60' 
                        : conversions[key]
                        ? 'bg-blue-50 border-blue-300'
                        : 'bg-white border-gray-300 hover:border-gray-400'
                    }`}>
                      <input 
                        type="checkbox"
                        checked={conversions[key]}
                        onChange={(e) => setConversions({ ...conversions, [key]: e.target.checked })}
                        disabled={isRunning || isDisabled}
                        className="mt-1 disabled:opacity-50"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-900 text-sm">{label}</p>
                        <p className="text-xs text-gray-600">{desc}</p>
                      </div>
                    </label>
                  );
                })}
              </div>

              {isRunning && (
                <div className="mt-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-700">Progress</span>
                    <span className="text-sm font-semibold text-gray-900">{Math.round(progress)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              <button
                onClick={handleRunConversion}
                disabled={isRunning}
                className="mt-6 w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold py-3 rounded-lg transition flex items-center justify-center gap-2"
              >
                <Play size={20} />
                {isRunning ? 'Running Conversion...' : 'Start Conversion'}
              </button>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Conversion Log</h2>
              
              <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm h-80 overflow-y-auto space-y-1">
                {logs.length === 0 ? (
                  <div className="text-gray-500">Logs will appear here...</div>
                ) : (
                  logs.map((log, idx) => (
                    <div key={idx} className={`flex gap-2 ${getLogColor(log.type)}`}>
                      <span className="text-gray-500 flex-shrink-0">[{log.timestamp}]</span>
                      <span className="flex-1">{log.message}</span>
                    </div>
                  ))
                )}
                <div ref={logsEndRef} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}