import { useState } from 'react';
import { Card, Form, Input, Button, Tabs, message, Typography, Space } from 'antd';
import { UserOutlined, LockOutlined, SmileOutlined, CameraOutlined, ToolOutlined } from '@ant-design/icons';
import { login, register } from '../api';

const { Title } = Typography;

export default function Login({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  const handleSubmit = async (values) => {
    setLoading(true);
    try {
      if (isLogin) {
        const res = await login({ username: values.username, password: values.password });
        if (res.data.code === 200) {
          localStorage.setItem('user', JSON.stringify(res.data.data));
          message.success('登录成功，欢迎 ' + res.data.data.nickname);
          onLogin?.(res.data.data);
        } else {
          message.error(res.data.data || '登录失败');
        }
      } else {
        const res = await register({
          username: values.username,
          nickname: values.nickname,
          password: values.password
        });
        if (res.data.code === 200) {
          message.success('注册成功，请登录');
          setIsLogin(true);
          form.resetFields(['password']);
        } else {
          message.error(res.data.data || '注册失败');
        }
      }
    } catch (err) {
      message.error('网络错误，请检查后端服务');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1890ff 0%, #001529 100%)',
      padding: '24px',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* 背景装饰 */}
      <div style={{
        position: 'absolute',
        top: '-10%',
        right: '-10%',
        width: '40%',
        height: '40%',
        background: 'rgba(255, 255, 255, 0.05)',
        borderRadius: '50%',
        filter: 'blur(80px)',
        zIndex: 0
      }} />
      <div style={{
        position: 'absolute',
        bottom: '-10%',
        left: '-10%',
        width: '50%',
        height: '50%',
        background: 'rgba(24, 144, 255, 0.1)',
        borderRadius: '50%',
        filter: 'blur(100px)',
        zIndex: 0
      }} />

      <div style={{
        display: 'flex',
        gap: '60px',
        maxWidth: 1100,
        width: '100%',
        flexWrap: 'wrap',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 1
      }}>

        {/* 左侧介绍 */}
        <div style={{ flex: '1 1 400px', maxWidth: 550, padding: '20px 0' }}>
          <Title level={1} style={{ color: '#fff', marginBottom: 24, fontSize: '42px', fontWeight: 700, textShadow: '0 2px 4px rgba(0,0,0,0.2)' }}>
            校园基础设施<br />智能巡检系统
          </Title>
          <p style={{ fontSize: 18, color: 'rgba(255,255,255,0.85)', lineHeight: 1.8, marginBottom: 40 }}>
            利用深度学习技术，为校园提供 24/7 全天候数字化巡检方案。<br />
            自动识别道路病害、设施损毁，让校园维护更智能、更高效。
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '20px' }}>
            <Card
              size="small"
              style={{
                borderRadius: 16,
                backgroundColor: 'rgba(255,255,255,0.1)',
                border: '1px solid rgba(255,255,255,0.2)',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}
              bodyStyle={{ padding: '20px' }}
            >
              <Space direction="vertical" size={12}>
                <div style={{
                  width: 48,
                  height: 48,
                  borderRadius: 12,
                  background: 'rgba(255,255,255,0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <CameraOutlined style={{ fontSize: 24, color: '#fff' }} />
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: '#fff', fontSize: 16 }}>智能识别</div>
                  <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: 13, marginTop: 4 }}>YOLO 实时病害检测</div>
                </div>
              </Space>
            </Card>
            <Card
              size="small"
              style={{
                borderRadius: 16,
                backgroundColor: 'rgba(255,255,255,0.1)',
                border: '1px solid rgba(255,255,255,0.2)',
                backdropFilter: 'blur(10px)',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}
              bodyStyle={{ padding: '20px' }}
            >
              <Space direction="vertical" size={12}>
                <div style={{
                  width: 48,
                  height: 48,
                  borderRadius: 12,
                  background: 'rgba(255,255,255,0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <ToolOutlined style={{ fontSize: 24, color: '#fff' }} />
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: '#fff', fontSize: 16 }}>闭环管理</div>
                  <div style={{ color: 'rgba(255,255,255,0.6)', fontSize: 13, marginTop: 4 }}>工单全流程追踪</div>
                </div>
              </Space>
            </Card>
          </div>
        </div>

        {/* 右侧登录卡片 */}
        <div style={{ flex: '0 1 400px', width: 400 }}>
          <Card
            style={{
              borderRadius: 24,
              boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
              overflow: 'hidden',
              border: 'none',
              backdropFilter: 'blur(20px)',
              background: 'rgba(255, 255, 255, 0.95)'
            }}
            bodyStyle={{ padding: '40px 32px' }}
          >
            <div style={{ textAlign: 'center', marginBottom: 32 }}>
              <Title level={3} style={{ margin: 0, fontWeight: 700 }}>{isLogin ? '欢迎回来' : '创建账号'}</Title>
              <p style={{ color: '#8c8c8c', marginTop: 8 }}>{isLogin ? '请登录您的管理后台' : '开始管理您的巡检任务'}</p>
            </div>

            <Tabs
              activeKey={isLogin ? 'login' : 'register'}
              onChange={(key) => {
                setIsLogin(key === 'login');
                form.resetFields();
              }}
              centered
              indicatorSize={(origin) => origin - 16}
              items={[
                {
                  key: 'login',
                  label: '登录',
                  children: (
                    <Form form={form} onFinish={handleSubmit} layout="vertical">
                      <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
                        <Input prefix={<UserOutlined />} placeholder="用户名" size="large" />
                      </Form.Item>
                      <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
                        <Input.Password prefix={<LockOutlined />} placeholder="密码" size="large" />
                      </Form.Item>
                      <Form.Item>
                        <Button type="primary" htmlType="submit" loading={loading} block size="large">
                          登录
                        </Button>
                      </Form.Item>
                    </Form>
                  ),
                },
                {
                  key: 'register',
                  label: '注册',
                  children: (
                    <Form form={form} onFinish={handleSubmit} layout="vertical">
                      <Form.Item name="username" rules={[{ required: true, message: '请输入用户名' }]}>
                        <Input prefix={<UserOutlined />} placeholder="用户名" size="large" />
                      </Form.Item>
                      <Form.Item name="nickname" rules={[{ required: true, message: '请输入昵称' }]}>
                        <Input prefix={<SmileOutlined />} placeholder="昵称" size="large" />
                      </Form.Item>
                      <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
                        <Input.Password prefix={<LockOutlined />} placeholder="密码" size="large" />
                      </Form.Item>
                      <Form.Item>
                        <Button type="primary" htmlType="submit" loading={loading} block size="large">
                          注册
                        </Button>
                      </Form.Item>
                    </Form>
                  ),
                },
              ]}
            />
          </Card>
        </div>
      </div>
    </div>
  );
}