import React, { useState, useEffect, useCallback } from 'react';
import { Layout, Typography, message, Spin } from 'antd';
import RepairList from './components/RepairList';
import UploadDetection from './components/UploadDetection';
import { getAllRepairs } from './api';
import 'antd/dist/reset.css';
import './App.css';
const { Header, Content, Footer } = Layout;
const { Title } = Typography;
const App = () => {
  const [repairs, setRepairs] = useState([]);
  const [loading, setLoading] = useState(true);
  const fetchTasks = useCallback(async () => {
    setLoading(true);
    try {
      const response = await getAllRepairs();
      setRepairs(response.data.data || response.data);
    } catch (error) {
      console.error('获取维修任务列表失败:', error);
      message.error('无法加载维修任务列表，请检查后端服务是否开启。');
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    fetchTasks();
  }, [fetchTasks]);
  return (
    <Layout className="layout" style={{ minHeight: '100vh' }}>
      <Header>
        <Title level={3} style={{ color: 'white', lineHeight: '64px', float: 'left', margin: 0 }}>
          校园基础设施智能巡检系统 - 管理面板
        </Title>
      </Header>
      <Content style={{ padding: '24px 50px' }}>
        <div className="site-layout-content" style={{ background: '#fff', padding: 24, borderRadius: 8 }}>
          <UploadDetection onTaskAdded={fetchTasks} />
          <Spin spinning={loading} tip="正在加载任务列表...">
            <RepairList repairs={repairs} loading={loading} refreshTasks={fetchTasks} />
          </Spin>
        </div>
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        AI Campus Inspection 2026 Created by AI Assistant
      </Footer>
    </Layout>
  );
};
export default App;
