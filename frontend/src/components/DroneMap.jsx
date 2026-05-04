import { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents } from 'react-leaflet';
import { Button, Input, message, Space } from 'antd';
import { PlusOutlined, EnvironmentOutlined } from '@ant-design/icons';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// 修复 Leaflet 图标
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// 默认信阳位置
const DEFAULT_CENTER = [32.0, 114.0]; // 信阳
const DEFAULT_ZOOM = 12;

const DroneMap = ({ onPointsChange, initialPoints = [] }) => {
  const [points, setPoints] = useState(initialPoints);
  
  // 点击地图添加点
  const MapClick = () => {
    useMapEvents({
      click: (e) => {
        const newPoint = {
          lat: Number(e.latlng.lat.toFixed(6)),
          lng: Number(e.latlng.lng.toFixed(6)),
        };
        const newPoints = [...points, newPoint];
        setPoints(newPoints);
        onPointsChange?.(newPoints);
      },
    });
    return null;
  };
  
  const removePoint = (index, e) => {
    e.stopPropagation(); // 阻止冒泡，避免触发 map 点击
    const newPoints = points.filter((_, i) => i !== index);
    setPoints(newPoints);
    onPointsChange?.(newPoints);
  };
  
  return (
    <div style={{ height: '400px', width: '100%', borderRadius: 8, overflow: 'hidden' }}>
      <MapContainer 
        center={DEFAULT_CENTER} 
        zoom={DEFAULT_ZOOM} 
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        <MapClick />
        {points.map((p, i) => (
          <Marker key={i} position={[p.lat, p.lng]}>
            <Popup>
              <Space direction="vertical">
                <div>点 {i + 1}: {p.lat.toFixed(6)}, {p.lng.toFixed(6)}</div>
                <Button size="small" danger onClick={(e) => removePoint(i, e)}>删除</Button>
              </Space>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      
      <div style={{ marginTop: 8, color: '#666' }}>
        点击地图添加航点 | 共 {points.length} 个点
        {points.length > 0 && (
          <pre style={{ background: '#f5f5f5', padding: 8, marginTop: 4, fontSize: 12 }}>
            {JSON.stringify(points)}
          </pre>
        )}
      </div>
    </div>
  );
};

export default DroneMap;