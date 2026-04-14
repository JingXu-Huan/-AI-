import React, { useState } from 'react';
import { Table, Button, Tag, Space, Modal, Select, message, Popconfirm, Image } from 'antd';
import { updateRepairStatus, deleteRepair, analyze, getRepairImgUrls } from '../api';

const { Option } = Select;

const statusColors = {
  pending: 'orange',
  processing: 'blue',
  completed: 'green',
  cancelled: 'grey',
};

const RepairList = ({ repairs, loading, refreshTasks }) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedRepair, setSelectedRepair] = useState(null);
  const [aiReport, setAiReport] = useState('');
  const [isReportLoading, setIsReportLoading] = useState(false);
  const [imgUrls, setImgUrls] = useState([]);

  const handleUpdateStatus = async (id, newStatus) => {
    try {
      await updateRepairStatus(id, newStatus);
      message.success(`任务 #${id} 状态已更新为 ${newStatus}`);
      refreshTasks();
    } catch (error) {
      console.error('更新状态失败:', error);
      message.error('更新状态失败');
    }
  };

  const handleDelete = async (id) => {
    try {
      await deleteRepair(id);
      message.success(`任务 #${id} 已被删除`);
      refreshTasks();
    } catch (error) {
      console.error('删除任务失败:', error);
      message.error('删除任务失败');
    }
  };

  const showDetails = async (record) => {
    setSelectedRepair(record);
    setAiReport('');
    setIsModalVisible(true);
    setImgUrls([]);
    try {
      const res = await getRepairImgUrls(record.id);
      setImgUrls(res.data.data || []);
    } catch (e) {
      console.error('获取图片失败:', e);
    }
  };

  const handleGenerateReport = async () => {
    if (!selectedRepair) return;
    setIsReportLoading(true);
    try {
      const aggregated = parseReportToDetectionResult(selectedRepair.report, selectedRepair.type);
      aggregated.location = selectedRepair.location;
      aggregated.id = selectedRepair.id;
      const result = await analyze(aggregated);
      // AiAnalysisResult is an object, convert to formatted string
      const reportData = result.data.data;
      const formatted = JSON.stringify(reportData, null, 2);
      setAiReport(formatted);
    } catch (error) {
      console.error('生成报告失败:', error);
      message.error('生成AI报告失败');
    } finally {
      setIsReportLoading(false);
    }
  };

  // 解析 report 字符串为 DetectionResult 格式
  const parseReportToDetectionResult = (report, type) => {
    const result = {
      meanConfidenceOfPothole: 0, meanConfidenceOfCrack: 0, meanConfidenceOfManhole: 0,
      meanConfidenceOfPatchNet: 0, meanConfidenceOfPatchCrack: 0, meanConfidenceOfPatchPothole: 0,
      lowPotholeCount: 0, lowCrackCount: 0, lowManholeCount: 0, lowPatchNetCount: 0, lowPatchCrackCount: 0, lowPatchPotholeCount: 0,
      mediumPotholeCount: 0, mediumCrackCount: 0, mediumManholeCount: 0, mediumPatchNetCount: 0, mediumPatchCrackCount: 0, mediumPatchPotholeCount: 0,
      highPotholeCount: 0, highCrackCount: 0, highManholeCount: 0, highPatchNetCount: 0, highPatchCrackCount: 0, highPatchPotholeCount: 0,
    };
    
    if (!report) return result;
    
    // 解析总数:52, 高:7, 中:29, 低:16
    const totalMatch = report.match(/总数:(\d+)/);
    const highMatch = report.match(/高:(\d+)/);
    const mediumMatch = report.match(/中:(\d+)/);
    const lowMatch = report.match(/低:(\d+)/);
    
    // 解析平均置信度
    const confMatches = report.match(/([A-Za-z]+)=([\d.]+)/g);
    if (confMatches) {
      confMatches.forEach(m => {
        const match = m.match(/([A-Za-z]+)=([\d.]+)/);
        if (match) {
          const [, typeName, conf] = match;
          const key = 'meanConfidenceOf' + typeName.charAt(0).toUpperCase() + typeName.slice(1).toLowerCase();
          if (result.hasOwnProperty(key)) {
            result[key] = parseFloat(conf);
          }
        }
      });
    }
    
    // 根据损坏类型设置对应计数
    const total = totalMatch ? parseInt(totalMatch[1]) : 0;
    const high = highMatch ? parseInt(highMatch[1]) : 0;
    const medium = mediumMatch ? parseInt(mediumMatch[1]) : 0;
    const low = lowMatch ? parseInt(lowMatch[1]) : 0;
    
    if (type && type.includes('Crack')) {
      result.highCrackCount = high;
      result.mediumCrackCount = medium;
      result.lowCrackCount = low;
    } else if (type && type.includes('Pothole')) {
      result.highPotholeCount = high;
      result.mediumPotholeCount = medium;
      result.lowPotholeCount = low;
    } else if (type && type.includes('Manhole')) {
      result.highManholeCount = high;
      result.mediumManholeCount = medium;
      result.lowManholeCount = low;
    } else if (type && type.includes('PatchNet')) {
      result.highPatchNetCount = high;
      result.mediumPatchNetCount = medium;
      result.lowPatchNetCount = low;
    } else if (type && type.includes('PatchCrack')) {
      result.highPatchCrackCount = high;
      result.mediumPatchCrackCount = medium;
      result.lowPatchCrackCount = low;
    } else if (type && type.includes('PatchPothole')) {
      result.highPatchPotholeCount = high;
      result.mediumPatchPotholeCount = medium;
      result.lowPatchPotholeCount = low;
    }
    
    return result;
  };

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', sorter: (a, b) => a.id - b.id },
    { title: '损坏类型', dataIndex: 'type', key: 'type' },
    { title: '位置', dataIndex: 'location', key: 'location' },
    {
      title: '图片',
      key: 'image',
      render: (_, record) => {
        // 表格中显示第一张图片（点击详情可看全部）
        const firstUrl = record.imageUrls?.[0];
        return firstUrl ? <Image src={firstUrl} width={60} height={60} style={{objectFit: 'cover', borderRadius: 4}} /> : <Tag>无</Tag>;
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => <Tag color={statusColors[status] || 'default'}>{status.toUpperCase()}</Tag>,
      filters: [
        { text: '待处理 (Pending)', value: 'pending' },
        { text: '处理中 (Processing)', value: 'processing' },
        { text: '已完成 (Completed)', value: 'completed' },
        { text: '已取消 (Cancelled)', value: 'cancelled' },
      ],
      onFilter: (value, record) => record.status.indexOf(value) === 0,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record) => (
        <Space size="middle">
          <Button type="link" onClick={() => showDetails(record)}>详情</Button>
          <Select
            defaultValue={record.status}
            style={{ width: 120 }}
            onChange={(newStatus) => handleUpdateStatus(record.id, newStatus)}
            size="small"
          >
            <Option value="pending">待处理</Option>
            <Option value="processing">处理中</Option>
            <Option value="completed">已完成</Option>
            <Option value="cancelled">取消</Option>
          </Select>
          <Popconfirm
            title="确定要删除这个任务吗?"
            onConfirm={() => handleDelete(record.id)}
            okText="是"
            cancelText="否"
          >
            <Button type="link" danger>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <>
      <Table
        columns={columns}
        dataSource={repairs}
        loading={loading}
        rowKey="id"
        pagination={{ pageSize: 10 }}
      />
      <Modal
        title={`维修任务详情 (ID: ${selectedRepair?.id})`}
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        footer={[
          <Button key="back" onClick={() => setIsModalVisible(false)}>
            关闭
          </Button>,
        ]}
      >
        {selectedRepair && (
          <div>
            <p><strong>损坏类型:</strong> {selectedRepair.type}</p>
            <p><strong>位置:</strong> {selectedRepair.location}</p>
            <p><strong>当前状态:</strong> <Tag color={statusColors[selectedRepair.status]}>{selectedRepair.status.toUpperCase()}</Tag></p>
            {imgUrls.length > 0 && (
              <p>
                <strong>检测图片:</strong><br />
                {imgUrls.map((url, idx) => (
                  <Image key={idx} src={url} style={{maxWidth: '100%', maxHeight: 300, marginTop: 8, marginRight: 8}} />
                ))}
              </p>
            )}
            <hr />
            <h4>AI维修报告生成</h4>
            <Button onClick={handleGenerateReport} loading={isReportLoading} style={{ marginBottom: 16 }}>
              请求AI生成维修建议
            </Button>
            {isReportLoading && <p>AI正在分析中，请稍候...</p>}
            {aiReport && <div style={{ background: '#f0f2f5', padding: '12px', borderRadius: '4px' }}><pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{aiReport}</pre></div>}
          </div>
        )}
      </Modal>
    </>
  );
};

export default RepairList;

