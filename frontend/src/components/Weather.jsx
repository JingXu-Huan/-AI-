import React, { useState, useEffect } from 'react';
import { Typography } from 'antd';

const { Text } = Typography;

const Weather = () => {
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchWeather = async () => {
      try {
        // 使用 wttr.in JSON 格式获取信阳天气
        const res = await fetch('https://wttr.in/Xinyang?format=j1');
        const json = await res.json();
        const curr = json.current_condition?.[0];
        if (curr) {
          const temp = curr.temp_C;
          const humidity = curr.humidity;
          const weatherCode = curr.weatherCode;
          // 天气代码映射
          const weatherEmoji = {
            '113': '☀️', '116': '⛅', '119': '☁️', '122': '🌫️',
            '176': '🌧️', '179': '🌨️', '182': '🌧️', '185': '🌨️',
            '200': '⛈️', '227': '🌨️', '230': '🌨️', '248': '🌫️',
            '260': '🌫️', '263': '🌧️', '266': '🌧️', '281': '🌧️',
            '284': '🌧️', '293': '🌧️', '296': '🌧️', '299': '🌧️',
            '302': '🌧️', '305': '🌧️', '308': '🌧️', '311': '🌧️',
            '314': '🌧️', '317': '🌧️', '320': '🌨️', '323': '🌨️',
            '326': '🌨️', '329': '🌨️', '332': '🌨️', '335': '🌨️',
            '338': '🌨️', '350': '🌧️', '353': '🌧️', '356': '🌧️',
            '359': '🌧️', '362': '🌧️', '365': '🌧️', '368': '🌧️',
            '371': '🌨️', '373': '🌧️', '374': '🌨️', '376': '🌧️',
            '379': '🌨️', '386': '⛈️', '389': '⛈️', '392': '⛈️', '395': '⛈️',
          };
          setWeather(`${weatherEmoji[weatherCode] || '🌤️'} ${temp}°C ${humidity}%`);
        } else {
          setWeather('获取天气失败');
        }
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
    <div style={{ display: 'inline-block', lineHeight: '64px' }}>
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