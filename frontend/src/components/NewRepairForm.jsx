import React, { useState } from 'react';
import { Button, Form, Input, Select, message, Card } from 'antd';
import { writeToDB } from '../api';
const { Option } = Select;
const NewRepairForm = ({ onTaskAdded }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const handleSubmit = async (values) => {
    setLoading(true);
    try {
      const detectionResult = {
        type: values.type,
        location: values.location,
      };
      await writeToDB(detectionResult);
      message.success('新维修任务已成功上报！');
      form.resetFields();
      if (onTaskAdded) {
        onTaskAdded(); 
      }
    } catch (error) {
      console.error('上报任务失败:', error);
      message.error('上报任务失败，请检查网络或联系管理员。');
    } finally {
      setLoading(false);
    }
  };
  return (
    <Card title="模拟新检测并上报维修任务" style={{ marginBottom: 24 }}>
      <Form
        form={form}
        layout="inline"
        onFinish={handleSubmit}
        initialValues={{ type: 'Crack', location: 'A区-教学楼' }}
      >
        <Form.Item
          name="type"
          label="损坏类型"
          rules={[{ required: true, message: '请选择损坏类型' }]}
        >
          <Select style={{ width: 150 }} placeholder="选择类型">
            <Option value="Crack">裂缝 (Crack)</Option>
            <Option value="Manhole">井盖 (Manhole)</Option>
            <Option value="Net">网裂 (Net)</Option>
            <Option value="Pothole">坑洞 (Pothole)</Option>
            <Option value="Patch-Crack">修补裂缝</Option>
            <Option value="Patch-Net">修补网裂</Option>
            <Option value="Patch-Pothole">修补坑洞</Option>
            <Option value="other">其他 (other)</Option>
          </Select>
        </Form.Item>
        <Form.Item
          name="location"
          label="发生位置"
          rules={[{ required: true, message: '请输入发生位置' }]}
        >
          <Input style={{ width: 200 }} placeholder="例如：A区-3号楼" />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" loading={loading}>
            上报任务
          </Button>
        </Form.Item>
      </Form>
    </Card>
  );
};
export default NewRepairForm;
