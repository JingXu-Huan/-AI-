import React, { useState } from 'react';
import { Upload, Button, Form, Input, message, Card, Table, Tag, Tabs } from 'antd';
import { UploadOutlined, SendOutlined } from '@ant-design/icons';
import { writeToDB, getDescFromAI } from '../api';
import './UploadDetection.css';

const UploadDetection = ({ onTaskAdded }) => {
  const [form] = Form.useForm();
  const [streamForm] = Form.useForm();
  const [fileList, setFileList] = useState([]);
  const [detecting, setDetecting] = useState(false);
  const [isLive, setIsLive] = useState(false);
  const [liveStreamUrl, setLiveStreamUrl] = useState(null);
  const [detectionResults, setDetectionResults] = useState([]);
  const [rawDetections, setRawDetections] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [imgUrl, setImgUrl] = useState(null);
  const [isVideo, setIsVideo] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);
  const [fetchingAI, setFetchingAI] = useState({});
  const [activeTab, setActiveTab] = useState('file');

  const runDetection = async (formData, onSuccess) => {
    setDetecting(true);
    setDetectionResults([]);
    setRawDetections(null);
    setSelectedRowKeys([]);
    setImgUrl(null);
    setIsVideo(false);

    try {
      const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText);
      }
      const result = await response.json();
      if (result.error) {
          throw new Error(result.error);
      }
      let detections = result.data;
      setRawDetections(detections);
      const isInvalid = !detections || !Array.isArray(detections) || detections.length === 0;
      if (isInvalid) {
          message.warning('检测完成，但未发现损伤目标或无数据返回！');
          if (onSuccess) onSuccess({}, null);
          setDetecting(false);
          return;
      }
      const isVideoOutput = detections[0].frame_index !== undefined;
      setIsVideo(isVideoOutput);
      if (isVideoOutput) {
         let allDets = [];
         for(let i=0; i<detections.length; i++) {
            if (detections[i].detections) {
                allDets = allDets.concat(detections[i].detections);
            }
         }
         detections = allDets;
      }
      if (detections.length === 0) {
          message.warning('检测完成，但未发现损伤目标！');
          if (onSuccess) onSuccess({}, null);
          setDetecting(false);
          return;
      }
      message.success('AI 检测成功！发现了 ' + detections.length + ' 个目标。');
      const formattedResults = [];
      let loc = formData.get('location') || '未知区域';
      for (let i = 0; i < detections.length; i++) {
        const det = detections[i];
        let typeName = 'Unknown';
        if (det.type) typeName = det.type;
        else if (det.class_name) typeName = det.class_name;
        let severityValue = 'low';
        if (det.severity) severityValue = det.severity;
        let bboxStr = 'N/A';
        if (det.bounding_box) {
           bboxStr = '[' + det.bounding_box.x1 + ', ' + det.bounding_box.y1 + ', ' + det.bounding_box.x2 + ', ' + det.bounding_box.y2 + ']';
        }
        const itemLoc = det.location || loc;
        formattedResults.push({
          key: i,
          type: typeName,
          confidence: det.confidence,
          location: itemLoc,
          severity: severityValue,
          bbox: bboxStr,
          report: null
        });
      }
      setDetectionResults(formattedResults);
      setSelectedRowKeys(formattedResults.map(item => item.key));
      if (result.stem) {
        setImgUrl('/api/image?stem=' + encodeURIComponent(result.stem));
      }
      message.success('检测分析完成！');
      if (onSuccess) onSuccess(result, null);
    } catch (error) {
      console.error(error);
      message.error('检测失败: ' + error.message);
    } finally {
      setDetecting(false);
      setIsLive(false);
      setLiveStreamUrl(null);
    }
  };

  const customRequest = async ({ file, onSuccess, onError }) => {
    const formData = new FormData();
    try {
      const values = await form.validateFields();
      formData.append('file', file);
      if (values && values.location) {
         formData.append('location', values.location);
      }
      await runDetection(formData, onSuccess);
    } catch (e) {
      onError(e);
    }
  };

  const handleLiveStart = async (values) => {
    setDetecting(true);
    setIsLive(true);
    setDetectionResults([]);
    setRawDetections(null);
    setImgUrl(null);
    const url = `/api/stream_live?source=${encodeURIComponent(values.stream_url)}&location=${encodeURIComponent(values.location)}`;
    setLiveStreamUrl(url);
  };

  const handleLiveStop = async () => {
     setDetecting(false);
     setIsLive(false);
     setLiveStreamUrl(null);
     
     try {
       const res = await fetch('/api/stream_stop');
       const result = await res.json();
       let detections = result.data || [];
       setRawDetections(detections);
       
       let allDets = [];
       for(let i=0; i<detections.length; i++) {
          if (detections[i].detections) {
              allDets = allDets.concat(detections[i].detections);
          }
       }
       detections = allDets;

       const formattedResults = [];
       for (let i = 0; i < detections.length; i++) {
          const det = detections[i];
          let typeName = 'Unknown';
          if (det.type) typeName = det.type;
          else if (det.class_name) typeName = det.class_name;
          
          formattedResults.push({
            key: i,
            type: typeName,
            confidence: det.confidence,
            location: det.location || '大门主干道',
            severity: det.severity || 'low',
            bbox: det.bounding_box ? `[${det.bounding_box.x1}, ${det.bounding_box.y1}, ${det.bounding_box.x2}, ${det.bounding_box.y2}]` : 'N/A',
            report: null
          });
       }
       setDetectionResults(formattedResults);
       setSelectedRowKeys(formattedResults.map(item => item.key));
       message.success(`流检测结束，共捕获 ${formattedResults.length} 个受损记录！`);
     } catch(e) {
       console.error('Stop error:', e);
     }
  };

  const handleFetchAI = async (recordKey) => {
    const item = detectionResults.find(r => r.key === recordKey);
    if (!item) return;

    setFetchingAI(prev => ({ ...prev, [recordKey]: true }));
    try {
      const aiResponse = await getDescFromAI({ type: item.type, location: item.location });
      const aiReport = aiResponse.data.data || '无报告内容';
      setDetectionResults(prev => prev.map(r => r.key === recordKey ? { ...r, report: aiReport } : r));
      message.success('AI 维修建议已生成！');
    } catch (e) {
      console.error('获取AI描述失败', e);
      message.error('生成报告失败');
    } finally {
      setFetchingAI(prev => ({ ...prev, [recordKey]: false }));
    }
  };

  const handleFetchAllAI = async () => {
     const toFetch = detectionResults.filter(item => selectedRowKeys.includes(item.key) && !item.report);
     if(toFetch.length === 0) {
        message.warning('选中的项目已生成或没有选中任何项。');
        return;
     }
     for(const item of toFetch) {
         await handleFetchAI(item.key);
     }
  };

  const onSelectChange = (newSelectedRowKeys) => {
    setSelectedRowKeys(newSelectedRowKeys);
  };

  const rowSelection = {
    selectedRowKeys,
    onChange: onSelectChange,
  };

  const executePush = async (itemsToPush) => {
    if (itemsToPush.length === 0) {
      message.warning('没有可推送的项！');
      return;
    }
    setSubmitting(true);
    try {
      for (let i = 0; i < itemsToPush.length; i++) {
        const item = itemsToPush[i];
        const payload = {
           type: item.type,
           location: item.location,
           report: item.report,
           severity: item.severity
        };
        await writeToDB(payload);
      }
      message.success(`成功推送 ${itemsToPush.length} 个检测结果至系统后端！`);

      const remainingResults = detectionResults.filter(r => !itemsToPush.includes(r));
      setDetectionResults(remainingResults);
      setSelectedRowKeys(selectedRowKeys.filter(key => remainingResults.some(r => r.key === key)));

      if (remainingResults.length === 0) {
          setFileList([]);
          setImgUrl(null);
      }

      if (onTaskAdded) {
         onTaskAdded();
      }
    } catch (error) {
      console.error(error);
      message.error('推送数据库失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handlePushSelected = () => {
    const itemsToPush = detectionResults.filter(item => selectedRowKeys.includes(item.key));
    executePush(itemsToPush);
  };

  const handlePushHigh = () => {
    const itemsToPush = detectionResults.filter(item => item.severity && item.severity.toLowerCase() === 'high');
    executePush(itemsToPush);
  };

  const columns = [
    { title: '类型', dataIndex: 'type', key: 'type' },
    { title: '置信度', dataIndex: 'confidence', key: 'confidence' },
    { 
      title: '严重程度', 
      dataIndex: 'severity', 
      key: 'severity',
      render: (severity) => {
        let color = 'green';
        if (severity === 'high') color = 'red';
        if (severity === 'medium') color = 'orange';
        let displayStr = '';
        if (severity) displayStr = severity.toUpperCase();
        return <Tag color={color}>{displayStr}</Tag>;
      }
    },
    { title: '位置', dataIndex: 'location', key: 'location' },
    {
      title: 'AI 维修建议',
      dataIndex: 'report',
      key: 'report',
      render: (report, record) => {
        if (!report) {
           return <Button size="small" type="primary" ghost onClick={() => handleFetchAI(record.key)} loading={fetchingAI[record.key]}>自动生成方案</Button>;
        }
        try {
          let strReport = report;
          if (typeof strReport === 'string') {
            // Remove markdown code blocks if any
            strReport = strReport.replace(/```json\s*/, '').replace(/```\s*$/, '').trim();
          }
          const parsed = typeof strReport === 'string' ? JSON.parse(strReport) : strReport;
          if (parsed && typeof parsed === 'object') {
            return (
              <div style={{ fontSize: '13px', lineHeight: '1.6' }}>
                <div style={{ marginBottom: '4px' }}>
                  <Tag color="blue">{parsed.problemType || '未知问题'}</Tag>
                  <Tag color={parsed.priority === '高' ? 'red' : parsed.priority === '中' ? 'orange' : 'green'}>{parsed.priority || '未知'}</Tag>
                </div>
                <div style={{ marginBottom: '4px' }}><strong>🛠️ 原因分析:</strong> <span style={{ color: '#666' }}>{parsed.cause || '未知'}</span></div>
                <div style={{ marginBottom: '4px' }}><strong>✅ 维修方案:</strong> <span style={{ color: '#666' }}>{parsed.repairPlan || '未知'}</span></div>
                <div><strong>⏱️ 预计工期:</strong> <span style={{ color: '#666', marginRight: '16px' }}>{parsed.estimatedTime || '未知'}</span></div>
              </div>
            );
          }
        } catch (e) {
          // ignore, fallback to string
        }
        return <div style={{ fontSize: '12px', whiteSpace: 'pre-wrap' }}>{typeof report === 'string' ? report.replace(/```json\s*/, '').replace(/```\s*$/, '').trim() : report}</div>;
      }
    },
  ];

  return (
    <Card className="detect-card" bordered={false} title={<span className="detect-title">📸 AI 智能实机检测</span>}>
      <Tabs activeKey={activeTab} onChange={setActiveTab} centered className="detect-tabs">
        <Tabs.TabPane tab="📁 本地文件上传" key="file">
          <div className="tab-content-wrapper">
            <Form
              form={form}
              layout="vertical"
              initialValues={{ location: 'A区-3号楼' }}
            >
              <Form.Item
                name="location"
                label="预设位置标签"
                rules={[{ required: true, message: '请输入发生位置' }]}
              >
                <Input placeholder="例如：A区" size="large" />
              </Form.Item>
              <Form.Item label="选择检测图片 / 视频">
                <Upload.Dragger
                  customRequest={customRequest}
                  fileList={fileList}
                  onChange={({ fileList: newFileList }) => setFileList(newFileList.slice(-1))}
                  accept="image/*,video/*"
                  showUploadList={false}
                  className="custom-dragger"
                >
                  <p className="ant-upload-drag-icon">
                    <UploadOutlined />
                  </p>
                  <p className="ant-upload-text">点击或拖拽文件到此区域上传并检测</p>
                  <p className="ant-upload-hint">支持 jpg, png, mp4 等常见格式</p>
                </Upload.Dragger>
                {detecting && <div className="detect-spin">🚀 AI 模型正在飞速推理中，请稍候...</div>}
              </Form.Item>
            </Form>
          </div>
        </Tabs.TabPane>

        <Tabs.TabPane tab="🎥 摄像头 / RTPS 流" key="stream">
           <div className="tab-content-wrapper">
            <Form
              form={streamForm}
              layout="vertical"
              initialValues={{ location: '大门主干道', stream_url: '0', max_frames: 100 }}
              onFinish={handleLiveStart}
            >
              <Form.Item
                name="location"
                label="预设位置标签"
                rules={[{ required: true }]}
              >
                <Input placeholder="例如：南门" size="large" />
              </Form.Item>
              <Form.Item
                name="stream_url"
                label="流地址或摄像头ID"
                rules={[{ required: true }]}
              >
                <Input placeholder="默认输入 0 可打开本地摄像头，或填写 rtsp://..." size="large" />
              </Form.Item>
              <Form.Item
                name="max_frames"
                label="最大采流帧数（防止死循环）"
                rules={[{ required: true }]}
              >
                <Input type="number" min={5} size="large" />
              </Form.Item>
              <Form.Item>
                {!isLive ? (
                  <Button type="primary" htmlType="submit" size="large" block className="stream-btn">
                    📡 开始实时流推流分析
                  </Button>
                ) : (
                  <Button danger type="primary" size="large" block onClick={handleLiveStop} className="stream-btn">
                    🛑 结束推流并生成报告
                  </Button>
                )}
              </Form.Item>
            </Form>
          </div>
        </Tabs.TabPane>
      </Tabs>

      {liveStreamUrl && isLive && (
         <div className="results-section">
           <div className="media-preview-box">
             <div style={{color:'#1890ff', marginBottom: '8px'}}>🔴 直播流侦测中...</div>
             <img src={liveStreamUrl} alt="Live Stream" className="media-preview" />
           </div>
         </div>
      )}

      {!isLive && detectionResults.length > 0 && (
        <div className="results-section">
          {imgUrl && (
             <div className="media-preview-box">
               {isVideo ? (
                 <video src={imgUrl} controls autoPlay loop muted className="media-preview video-preview" />
               ) : (
                 <img src={imgUrl} alt="Annotated Output" className="media-preview" />
               )}
             </div>
          )}

          {rawDetections && (
             <div className="raw-json-box">
                <h4>🗃️ 原始 JSON 机器返回数据 <span style={{fontSize: '12px', fontWeight: 'normal', color: '#999'}}>——供二次开发和对接参考</span></h4>
                <div className="json-container">
                  <pre>
                    {JSON.stringify(rawDetections, null, 2)}
                  </pre>
                </div>
             </div>
          )}

          <div className="table-header">
            <h4>📊 缺陷定位分析报告 ({detectionResults.length} 处损坏)</h4>
          </div>

          <Table
            rowSelection={rowSelection}
            dataSource={detectionResults}
            columns={columns} 
            pagination={{ pageSize: 10 }}
            size="middle"
            className="results-table"
          />
          <div className="action-button-group">
            <Button
              type="default"
              size="large"
              className="action-btn batch-ai-btn"
              onClick={handleFetchAllAI}
              disabled={selectedRowKeys.length === 0}
            >
              🧩 一键为选中项生成 AI 建议
            </Button>
            <Button
              type="primary"
              size="large"
              className="action-btn"
              icon={<SendOutlined />}
              onClick={handlePushSelected}
              loading={submitting}
              disabled={selectedRowKeys.length === 0}
            >
              🚀 推送选中项至系统
            </Button>
            <Button
              danger
              type="primary"
              size="large"
              className="action-btn"
              icon={<SendOutlined />}
              onClick={handlePushHigh}
              loading={submitting}
              disabled={!detectionResults.some(item => item.severity && item.severity.toLowerCase() === 'high')}
            >
              🚨 一键归档严重(High)项
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
};
export default UploadDetection;
