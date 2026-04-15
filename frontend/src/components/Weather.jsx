import React, { useState, useEffect } from 'react';
import { Typography } from 'antd';

const { Text } = Typography;

const Weather = () => {
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        // 使用 wttr.in 获取信阳天气
        const res = await fetch('https://wttr.in/%E4%BF%A1%E9%98%B3?format=%c+%t+%h&lang=zh');
        const text = await res.text();
        setWeather(text);
      } catch (e) {
        setWeather('获取天气失败');
      } finally {
        setLoading(false);
      }
    };
    fetchWeather();
    // 每10分钟刷新一次
    const interval = setInterval(fetchWeather, 10 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: 'inline-block', marginLeft: '24px', lineHeight: '64px' }}>
      {loading ? (
        <Text style={{ color: '#fff' }}>🌤️ 加载中...</Text>
      ) : (
        <Text style={{ color: '#fff', fontSize: '14px' }}>
          📍 信阳 {weather || '🌤️'}
        </Text>
      )}
    </div>
  );
};

export default Weather;