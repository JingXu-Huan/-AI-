import React, { useState, useMemo } from 'react';
import { Upload, Button, Form, Input, InputNumber, message, Card, Tag, Tabs, Collapse, Row, Col, Statistic } from 'antd';
import { UploadOutlined, SendOutlined, LeftOutlined, RightOutlined, EnvironmentOutlined } from '@ant-design/icons';
import { writeToDB, analyze } from '../api';
import './UploadDetection.css';

// 聚合检测结果为 DetectionResult 格式
const aggregateResults = (detections) => {
  const result = {
    meanConfidenceOfPothole: 0, meanConfidenceOfCrack: 0, meanConfidenceOfManhole: 0,
    meanConfidenceOfPatchNet: 0, meanConfidenceOfPatchCrack: 0, meanConfidenceOfPatchPothole: 0,
    lowPotholeCount: 0, lowCrackCount: 0, lowManholeCount: 0, lowPatchNetCount: 0, lowPatchCrackCount: 0, lowPatchPotholeCount: 0,
    mediumPotholeCount: 0, mediumCrackCount: 0, mediumManholeCount: 0, mediumPatchNetCount: 0, mediumPatchCrackCount: 0, mediumPatchPotholeCount: 0,
    highPotholeCount: 0, highCrackCount: 0, highManholeCount: 0, highPatchNetCount: 0, highPatchCrackCount: 0, highPatchPotholeCount: 0,
  };
  
  const typeMap = { 
    'Pothole': 'Pothole', 'Crack': 'Crack', 'Manhole': 'Manhole', 
    'PatchNet': 'PatchNet', 'PatchCrack': 'PatchCrack', 'PatchPothole': 'PatchPothole',
    'Net': 'PatchNet', 'Patch-Crack': 'PatchCrack', 'Patch-Pothole': 'PatchPothole'
  };
  const severityMap = { 'low': 'low', 'medium': 'medium', 'high': 'high', 'LOW': 'low', 'MEDIUM': 'medium', 'HIGH': 'high' };
  
  let counts = { Pothole: { low: 0, medium: 0, high: 0, conf: 0 },
                 Crack: { low: 0, medium: 0, high: 0, conf: 0 },
                 Manhole: { low: 0, medium: 0, high: 0, conf: 0 },
                 PatchNet: { low: 0, medium: 0, high: 0, conf: 0 },
                 PatchCrack: { low: 0, medium: 0, high: 0, conf: 0 },
                 PatchPothole: { low: 0, medium: 0, high: 0, conf: 0 } };
  
  detections.forEach(det => {
    const type = typeMap[det.type] || det.type;
    const severity = severityMap[det.severity?.toUpperCase()] || 'low';
    if (counts[type]) {
      counts[type][severity]++;
      counts[type].conf += det.confidence || 0;
    }
  });
  
  Object.keys(counts).forEach(type => {
    const total = counts[type].low + counts[type].medium + counts[type].high;
    const avgConf = total > 0 ? counts[type].conf / total : 0;
    result[`meanConfidenceOf${type}`] = avgConf;
    result[`low${type}Count`] = counts[type].low;
    result[`medium${type}Count`] = counts[type].medium;
    result[`high${type}Count`] = counts[type].high;
  });
  
  return result;
};

const UploadDetection = ({ onTaskAdded }) => {
  const [form] = Form.useForm();
  const [streamForm] = Form.useForm();
  const [fileList, setFileList] = useState([]);
  const [isLive, setIsLive] = useState(false);
  const [liveStreamUrl, setLiveStreamUrl] = useState(null);
  const [detecting, setDetecting] = useState(false);
  const [detectionResults, setDetectionResults] = useState([]);
  const [rawDetections, setRawDetections] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [imgUrl, setImgUrl] = useState(null);
  const [isVideo, setIsVideo] = useState(false);
  const [frameList, setFrameList] = useState([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [fetchingAI, setFetchingAI] = useState({});
  const [activeTab, setActiveTab] = useState('file');
  const [aiReports, setAiReports] = useState({});

  // 按区域分组
  const groupedResults = useMemo(() => {
    const groups = {};
    detectionResults.forEach(det => {
      const loc = det.location || '未知区域';
      if (!groups[loc]) groups[loc] = [];
      groups[loc].push(det);
    });
    return groups;
  }, [detectionResults]);

  const runDetection = async (formData, onSuccess) => {
    setDetecting(true);
    setDetectionResults([]);
    setRawDetections(null);
    setAiReports({});
    setImgUrl(null);
    setIsVideo(false);

    try {
      const response = await fetch('/api/detect', { method: 'POST', body: formData });
      if (!response.ok) throw new Error(await response.text());
      const result = await response.json();
      if (result.error) throw new Error(result.error);
      
      let detections = result.data;
      setRawDetections(detections);
      if (!detections || !Array.isArray(detections) || detections.length === 0) {
        message.warning('检测完成，但未发现损伤目标！');
        setDetecting(false);
        return;
      }
      
      const isVideoOutput = detections[0].frame_index !== undefined;
      setIsVideo(isVideoOutput);
      if (isVideoOutput) {
        let allDets = [];
        detections.forEach(d => { if (d.detections) allDets = allDets.concat(d.detections); });
        detections = allDets;
      }
      
      if (detections.length === 0) {
        message.warning('检测完成，但未发现损伤目标！');
        setDetecting(false);
        return;
      }
      
      message.success('AI 检测成功！发现了 ' + detections.length + ' 个目标。');
      
      const formattedResults = [];
      let loc = formData.get('location') || '未知区域';
      detections.forEach((det, i) => {
        formattedResults.push({
          key: i,
          type: det.type || det.class_name || 'Unknown',
          confidence: det.confidence,
          location: det.location || loc,
          severity: (det.severity || 'low').toLowerCase(),
          bbox: det.bounding_box ? `[${det.bounding_box.x1}, ${det.bounding_box.y1}, ${det.bounding_box.x2}, ${det.bounding_box.y2}]` : 'N/A',
        });
      });
      
      setDetectionResults(formattedResults);
      if (result.stem) {
        setImgUrl('/api/image?stem=' + encodeURIComponent(result.stem));
        if (isVideoOutput) {
          try {
            const res = await fetch('/api/frames?stem=' + encodeURIComponent(result.stem));
            const data = await res.json();
            if (data.frames?.length > 0) { setFrameList(data.frames); setCurrentFrame(0); }
          } catch (e) { console.error('获取帧列表失败:', e); }
        }
      }
    } catch (error) {
      message.error('检测失败: ' + error.message);
    } finally {
      setDetecting(false);
    }
  };

  const customRequest = async ({ file, onSuccess, onError }) => {
    const formData = new FormData();
    try {
      const values = await form.validateFields();
      formData.append('file', file);
      if (values?.location) formData.append('location', values.location);
      if (values?.max_frames) formData.append('max_frames', values.max_frames);
      await runDetection(formData, onSuccess);
    } catch (e) { onError(e); }
  };

  const handleLiveStart = async (values) => {
    setDetecting(true); setIsLive(true); setDetectionResults([]); setRawDetections(null); setAiReports({}); setImgUrl(null);
    setLiveStreamUrl(`/api/stream_live?source=${encodeURIComponent(values.stream_url)}&location=${encodeURIComponent(values.location)}`);
  };

  const handleLiveStop = async () => {
    setDetecting(false); setIsLive(false); setLiveStreamUrl(null);
    try {
      const res = await fetch('/api/stream_stop');
      const result = await res.json();
      let detections = result.data || [];
      let allDets = [];
      detections.forEach(d => { if (d.detections) allDets = allDets.concat(d.detections); });
      detections = allDets;
      
      const formattedResults = detections.map((det, i) => ({
        key: i,
        type: det.type || det.class_name || 'Unknown',
        confidence: det.confidence,
        location: det.location || '大门主干道',
        severity: (det.severity || 'low').toLowerCase(),
        bbox: det.bounding_box ? `[${det.bounding_box.x1}, ${det.bounding_box.y1}, ${det.bounding_box.x2}, ${det.bounding_box.y2}]` : 'N/A',
      }));
      setDetectionResults(formattedResults);
      message.success(`流检测结束，共捕获 ${formattedResults.length} 个受损记录！`);
    } catch(e) { console.error('Stop error:', e); }
  };

  // 按区域生成AI建议
  const handleFetchAIByLocation = async (location) => {
    setFetchingAI(prev => ({ ...prev, [location]: true }));
    try {
      const items = groupedResults[location] || [];
      const aggregated = aggregateResults(items);
      // 无状态调用，使用 id=0 表示非关联到具体维修任务
      aggregated.location = location;
      aggregated.id = 0;
      const aiResponse = await analyze(aggregated);
      setAiReports(prev => ({ ...prev, [location]: aiResponse.data.data || {} }));
      message.success(`区域 "${location}" AI 维修建议已生成！`);
    } catch (e) { message.error('生成报告失败'); } 
    finally { setFetchingAI(prev => ({ ...prev, [location]: false })); }
  };

  // 获取当前图片作为 File 对象
  const getCurrentImageFile = async () => {
    if (!imgUrl) return null;
    try {
      const response = await fetch(imgUrl);
      const blob = await response.blob();
      // 从 imgUrl 提取文件名
      const stem = new URL(imgUrl, 'http://localhost').searchParams.get('stem') || 'image';
      const ext = stem.includes('_annotated') ? '.jpg' : '.jpg';
      const filename = stem + ext;
      return new File([blob], filename, { type: 'image/jpeg' });
    } catch (e) { return null; }
  };

  // 按区域推送
  const handlePushByLocation = async (location) => {
    const items = groupedResults[location] || [];
    if (!items.length) return;
    setSubmitting(true);
    try {
      const aggregated = aggregateResults(items);
      aggregated.location = location;
      // 直接附加图片文件
      const imageFile = await getCurrentImageFile();
      await writeToDB(aggregated, imageFile);
      message.success(`区域 "${location}" 检测结果已推送！`);
      setDetectionResults(prev => prev.filter(r => r.location !== location));
      setAiReports(prev => { const n = {...prev}; delete n[location]; return n; });
      if (onTaskAdded) onTaskAdded();
    } catch (e) { message.error('推送数据库失败'); }
    finally { setSubmitting(false); }
  };

  // 推送所有区域
  const handlePushAll = async () => {
    const locations = Object.keys(groupedResults);
    if (!locations.length) return;
    setSubmitting(true);
    try {
      const imageFile = await getCurrentImageFile();
      for (const location of locations) {
        const items = groupedResults[location] || [];
        const aggregated = aggregateResults(items);
        aggregated.location = location;
        await writeToDB(aggregated, imageFile);
      }
      message.success(`已推送 ${locations.length} 个区域的检测结果！`);
      setDetectionResults([]); setAiReports({}); setImgUrl(null);
      if (onTaskAdded) onTaskAdded();
    } catch (e) { message.error('推送数据库失败'); }
    finally { setSubmitting(false); }
  };

  // 为所有区域生成AI建议
  const handleFetchAllAI = async () => {
    const locations = Object.keys(groupedResults);
    if (!locations.length) return;
    setFetchingAI(prev => { const p = {}; locations.forEach(l => p[l] = true); return { ...prev, ...p }; });
    try {
      for (const location of locations) {
        const items = groupedResults[location] || [];
        const aggregated = aggregateResults(items);
        // 无状态调用（批量/区域分析），id=0 表示不关联到数据库中的某个任务
        aggregated.location = location;
        aggregated.id = 0;
        const aiResponse = await analyze(aggregated);
        setAiReports(prev => ({ ...prev, [location]: aiResponse.data.data || {} }));
      }
      message.success('所有区域 AI 维修建议已生成！');
    } catch (e) { message.error('生成报告失败'); }
    finally { setFetchingAI({}); }
  };

  const renderReport = (report) => {
    if (!report) return null;
    try {
      let strReport = typeof report === 'string' ? report.replace(/```json\s*/, '').replace(/```\s*$/, '').trim() : report;
      const parsed = typeof strReport === 'string' ? JSON.parse(strReport) : strReport;
      if (parsed && typeof parsed === 'object') {
        return (
          <div style={{ fontSize: '13px', lineHeight: '1.6' }}>
            <div style={{ marginBottom: '8px' }}>
              <Tag color="blue">{parsed.problemType || '未知问题'}</Tag>
              <Tag color={parsed.priority === '高' ? 'red' : parsed.priority === '中' ? 'orange' : 'green'}>{parsed.priority || '未知'}</Tag>
            </div>
            <div><strong>🛠️ 原因:</strong> <span style={{ color: '#666' }}>{parsed.cause || '未知'}</span></div>
            <div><strong>✅ 方案:</strong> <span style={{ color: '#666' }}>{parsed.repairPlan || '未知'}</span></div>
            <div><strong>⏱️ 工期:</strong> <span style={{ color: '#666' }}>{parsed.estimatedTime || '未知'}</span></div>
          </div>
        );
      }
    } catch (e) {}
    return <div style={{ fontSize: '12px' }}>{typeof report === 'string' ? report : JSON.stringify(report)}</div>;
  };

  return (
    <Card className="detect-card" bordered={false} title={
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="detect-title">📸 AI 智能实机检测</span>
        {detectionResults.length > 0 && !isLive && (
          <Button type="primary" onClick={() => { setDetectionResults([]); setImgUrl(null); setRawDetections(null); setAiReports({}); }}>
            🔙 返回
          </Button>
        )}
      </div>
    }>
      {!(detectionResults.length > 0 && !isLive) && (
      <Tabs activeKey={activeTab} onChange={setActiveTab} centered>
        <Tabs.TabPane tab="📁 本地文件上传" key="file">
          <Form form={form} layout="vertical" initialValues={{ location: 'A区-3号楼' }}>
            <Form.Item name="location" label="预设位置标签" rules={[{ required: true }]}>
              <Input placeholder="例如：A区" size="large" />
            </Form.Item>
            <Form.Item name="max_frames" label="最大处理帧数" tooltip="建议100-500">
              <InputNumber min={10} max={1000} defaultValue={200} style={{ width: '100%' }} />
            </Form.Item>
            <Form.Item label="选择检测图片 / 视频">
              <Upload.Dragger customRequest={customRequest} fileList={fileList} onChange={({ fileList: f }) => setFileList(f.slice(-1))} accept="image/*,video/*" showUploadList={false}>
                <p className="ant-upload-drag-icon"><UploadOutlined /></p>
                <p className="ant-upload-text">点击或拖拽文件上传并检测</p>
                <p className="ant-upload-hint">支持 jpg, png, mp4 等格式</p>
              </Upload.Dragger>
              {detecting && <div className="detect-spin">🚀 YOLO 模型正在推理中，请稍候...</div>}
            </Form.Item>
          </Form>
        </Tabs.TabPane>

        <Tabs.TabPane tab="🎥 摄像头 / RTSP 流" key="stream">
          <Form form={streamForm} layout="vertical" initialValues={{ location: '大门主干道', stream_url: '0', max_frames: 100 }} onFinish={handleLiveStart}>
            <Form.Item name="location" label="预设位置标签" rules={[{ required: true }]}>
              <Input size="large" />
            </Form.Item>
            <Form.Item name="stream_url" label="流地址或摄像头ID" rules={[{ required: true }]}>
              <Input placeholder="输入 0 打开本地摄像头，或 rtsp://..." size="large" />
            </Form.Item>
            <Form.Item name="max_frames" label="最大采流帧数" rules={[{ required: true }]}>
              <Input type="number" min={5} size="large" />
            </Form.Item>
            <Form.Item>
              {!isLive ? (
                <Button type="primary" htmlType="submit" size="large" block>📡 开始实时流分析</Button>
              ) : (
                <Button danger type="primary" size="large" block onClick={handleLiveStop}>🛑 结束推流</Button>
              )}
            </Form.Item>
          </Form>
        </Tabs.TabPane>
      </Tabs>
      )}

      {liveStreamUrl && isLive && (
        <div className="results-section">
          <div style={{color:'#1890ff', marginBottom: '8px'}}>🔴 直播流侦测中...</div>
          <img src={liveStreamUrl} alt="Live Stream" className="media-preview" style={{ maxHeight: '600px' }} />
        </div>
      )}

      {detectionResults.length > 0 && (
        <div className="results-section" style={{ marginTop: '24px' }}>
          <Row gutter={[24, 24]}>
            {imgUrl && (
              <Col xs={24} lg={10} xl={8}>
                <Collapse defaultActiveKey={['media']} items={[
                  { key: 'media', label: <h4>🖼️ 媒体检测画面</h4>, children: (
                    <div>
                      {isVideo && frameList.length > 0 ? (
                        <>
                          <img src={`${imgUrl}&frame=${frameList[currentFrame]?.replace('frame_','').replace('.jpg','')}`} alt={`帧`} className="media-preview" style={{ width: '100%', maxHeight: '400px' }} />
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '16px', marginTop: '16px' }}>
                            <Button icon={<LeftOutlined />} onClick={() => setCurrentFrame(Math.max(0, currentFrame - 1))} disabled={currentFrame < 1} />
                            <span>{currentFrame + 1} / {frameList.length}</span>
                            <Button icon={<RightOutlined />} onClick={() => setCurrentFrame(Math.min(frameList.length - 1, currentFrame + 1))} disabled={currentFrame >= frameList.length - 1} />
                          </div>
                        </>
                      ) : (
                        <img src={imgUrl} alt="Annotated" className="media-preview" style={{ width: '100%', maxHeight: '500px' }} />
                      )}
                    </div>
                  )}
                ]} />
                {rawDetections && (
                  <Collapse items={[{ key: 'json', label: <h4>🗃️ 原始 JSON</h4>, children: <pre style={{ maxHeight: '300px', overflow: 'auto', fontSize: '12px' }}>{JSON.stringify(rawDetections, null, 2)}</pre> }]} style={{ marginTop: '16px' }} />
                )}
              </Col>
            )}

            <Col xs={24} lg={imgUrl ? 14 : 24} xl={imgUrl ? 16 : 24}>
              <h4 style={{ marginBottom: '16px' }}>📊 缺陷分析 ({detectionResults.length} 处, {Object.keys(groupedResults).length} 个区域)</h4>
              
              <Collapse
                defaultActiveKey={Object.keys(groupedResults)}
                items={Object.entries(groupedResults).map(([location, items]) => {
                  const total = items.length;
                  const highCount = items.filter(i => i.severity === 'high').length;
                  const mediumCount = items.filter(i => i.severity === 'medium').length;
                  const lowCount = items.filter(i => i.severity === 'low').length;
                  const hasReport = !!aiReports[location];
                  
                  const typeCounts = items.reduce((acc, i) => { acc[i.type] = (acc[i.type] || 0) + 1; return acc; }, {});
                  const allTypes = Object.entries(typeCounts).sort((a, b) => b[1] - a[1]);
                  const topTypes = allTypes.slice(0, 4);
                  const otherTypes = allTypes.slice(4);
                  
                  const avgConfidence = items.reduce((sum, i) => sum + (i.confidence || 0), 0) / total;
                  const highConfCount = items.filter(i => (i.confidence || 0) > 0.8).length;
                  const midConfCount = items.filter(i => (i.confidence || 0) > 0.5 && (i.confidence || 0) <= 0.8).length;
                  const lowConfCount = items.filter(i => (i.confidence || 0) <= 0.5).length;
                  
                  const severityData = [
                    { label: '严重', count: highCount, color: '#ff4d4f' },
                    { label: '中等', count: mediumCount, color: '#fa8c16' },
                    { label: '轻微', count: lowCount, color: '#52c41a' }
                  ];
                  const maxCount = Math.max(...severityData.map(d => d.count), 1);
                  
                  return {
                    key: location,
                    label: (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                        <EnvironmentOutlined />
                        <span style={{ fontWeight: 'bold' }}>{location}</span>
                        <Tag color="blue">{total}处</Tag>
                        {highCount > 0 && <Tag color="red">紧急 {highCount}</Tag>}
                      </div>
                    ),
                    extra: (
                      <div style={{ display: 'flex', gap: '8px' }} onClick={e => e.stopPropagation()}>
                        {!hasReport && <Button size="small" type="primary" ghost onClick={() => handleFetchAIByLocation(location)} loading={fetchingAI[location]}>生成AI</Button>}
                        <Button size="small" type="primary" icon={<SendOutlined />} onClick={() => handlePushByLocation(location)} loading={submitting}>推送</Button>
                      </div>
                    ),
                    children: (
                      <div>
                        <div style={{ marginBottom: '16px' }}>
                          <div style={{ fontSize: '13px', marginBottom: '12px', fontWeight: 600 }}>🔴 严重程度分布</div>
                          <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-end', height: '160px', padding: '0 16px', marginBottom: '8px' }}>
                            {severityData.map(d => (
                              <div key={d.label} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                <div style={{ 
                                  width: '100%', 
                                  height: `${(d.count / maxCount) * 135}px`,
                                  background: `linear-gradient(to top, ${d.color}dd, ${d.color})`,
                                  borderRadius: '6px 6px 0 0',
                                  minHeight: d.count > 0 ? '6px' : '0',
                                  transition: 'height 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
                                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                                }} />
                                <div style={{ fontSize: '14px', marginTop: '6px', color: d.color, fontWeight: 800 }}>{d.count}</div>
                                <div style={{ fontSize: '12px', color: '#555', fontWeight: 500 }}>{d.label}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        <div style={{ display: 'flex', gap: '16px', marginBottom: '16px', padding: '16px', background: '#f8f9fa', borderRadius: '8px', border: '1px solid #f0f0f0' }}>
                          <div style={{ flex: 1, textAlign: 'center' }}>
                            <div style={{ fontSize: '20px', fontWeight: 600, color: '#1890ff' }}>{(avgConfidence * 100).toFixed(1)}%</div>
                            <div style={{ fontSize: '11px', color: '#888' }}>平均置信度</div>
                          </div>
                          <div style={{ width: '1px', background: '#ddd' }} />
                          <div style={{ flex: 1, textAlign: 'center' }}>
                            <div style={{ fontSize: '20px', fontWeight: 600, color: '#52c41a' }}>{highConfCount}</div>
                            <div style={{ fontSize: '11px', color: '#888' }}>高置信 (&gt;80%)</div>
                          </div>
                          <div style={{ width: '1px', background: '#ddd' }} />
                          <div style={{ flex: 1, textAlign: 'center' }}>
                            <div style={{ fontSize: '20px', fontWeight: 600, color: '#fa8c16' }}>{midConfCount}</div>
                            <div style={{ fontSize: '11px', color: '#888' }}>中置信 (50-80%)</div>
                          </div>
                          <div style={{ width: '1px', background: '#ddd' }} />
                          <div style={{ flex: 1, textAlign: 'center' }}>
                            <div style={{ fontSize: '20px', fontWeight: 600, color: '#999' }}>{lowConfCount}</div>
                            <div style={{ fontSize: '11px', color: '#888' }}>低置信 (&lt;50%)</div>
                          </div>
                        </div>
                        
                        <div style={{ fontSize: '12px', marginBottom: '8px', fontWeight: 500 }}>📋 缺陷类型</div>
                        <div style={{ fontSize: '12px', color: '#888', marginBottom: '8px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                          {topTypes.map(([type, count]) => (
                            <Tag key={type} color="blue">{type}: {count}</Tag>
                          ))}
                          {otherTypes.map(([type, count]) => (
                            <Tag key={type} color="default">{type}: {count}</Tag>
                          ))}
                        </div>
                        
                        {hasReport && <div style={{ marginTop: '12px', padding: '12px', background: '#f5f5f5', borderRadius: '6px' }}>{renderReport(aiReports[location])}</div>}
                      </div>
                    )
                  };
                })}
              />

              <div style={{ marginTop: '24px', display: 'flex', gap: '12px' }}>
                <Button type="default" size="large" onClick={handleFetchAllAI} disabled={!Object.keys(groupedResults).length}>🧩 生成所有AI建议</Button>
                <Button type="primary" size="large" icon={<SendOutlined />} onClick={handlePushAll} loading={submitting} disabled={!detectionResults.length}>🚀 推送所有区域</Button>
              </div>
            </Col>
          </Row>
        </div>
      )}
    </Card>
  );
};
export default UploadDetection;
