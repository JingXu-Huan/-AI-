import React, { useState } from 'react';
import { Table, Button, Tag, Space, Modal, Select, message, Popconfirm } from 'antd';
import { updateRepairStatus, deleteRepair, getDescFromAI } from '../api';

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

  const showDetails = (record) => {
    setSelectedRepair(record);
    setAiReport(''); // 清空旧报告
    setIsModalVisible(true);
  };

  const handleGenerateReport = async () => {
    if (!selectedRepair) return;
    setIsReportLoading(true);
    try {
      const result = await getDescFromAI({
        type: selectedRepair.type,
        location: selectedRepair.location,
      });
      setAiReport(result.data.data);
    } catch (error) {
      console.error('生成报告失败:', error);
      message.error('生成AI报告失败');
    } finally {
      setIsReportLoading(false);
    }
  };

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id', sorter: (a, b) => a.id - b.id },
    { title: '损坏类型', dataIndex: 'type', key: 'type' },
    { title: '位置', dataIndex: 'location', key: 'location' },
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

