import React, { useState, useEffect } from 'react';
import { Card, Table, Button, Tag, Modal, Form, InputNumber, message, Row, Col, Statistic, Space, Progress, Tooltip, Badge, Input, DatePicker, Empty } from 'antd';
import dayjs from 'dayjs';
import { PlusOutlined, RocketOutlined, ThunderboltOutlined, AimOutlined, SyncOutlined, DeleteOutlined, SendOutlined } from '@ant-design/icons';
import { getAllDrones, getTaskList, addTask, cancelTask, retryTask, addDrone, getTasksByDrone, getTaskImages } from '../api';
import DroneMap from './DroneMap';
import axios from 'axios';

const { RangePicker } = DatePicker;

// 状态颜色映射
const statusColors = {
  idle: 'default',
  standby: 'processing',
  flying: 'blue',
  returning: 'orange',
  landing: 'gold',
  charging: 'green',
  error: 'red',
};

const taskStatusColors = {
  queued: 'default',
  ready: 'processing',
  running: 'blue',
  finished: 'success',
  failed: 'error',
  cancelled: 'default',
};

const DroneTask = ({ compact = false }) => {
    const [drones, setDrones] = useState([]);
    const [tasks, setTasks] = useState([]);
    const [loading, setLoading] = useState(true);
    const [modalOpen, setModalOpen] = useState(false);
    const [retryModalOpen, setRetryModalOpen] = useState(false);
    const [droneModalOpen, setDroneModalOpen] = useState(false);
    const [droneTasksOpen, setDroneTasksOpen] = useState(false);
    const [retryTaskId, setRetryTaskId] = useState(null);
    const [selectedDroneNo, setSelectedDroneNo] = useState(null);
    const [droneTasks, setDroneTasks] = useState([]);
    const [form] = Form.useForm();
    const [retryForm] = Form.useForm();
    const [droneForm] = Form.useForm();
    const [mapPoints, setMapPoints] = useState([]);
    const [imageModalOpen, setImageModalOpen] = useState(false);
    const [taskImages, setTaskImages] = useState([]);
    const [imageLoading, setImageLoading] = useState(false);

    // 查看任务结果图片
    const showTaskImages = async (taskId) => {
        setImageLoading(true);
        setImageModalOpen(true);
        try {
            const res = await getTaskImages(taskId);
            const imageNames = res.data?.data || [];
            // 获取每张图片的Base64
            const imagesWithData = await Promise.all(
                imageNames.map(async (name) => {
                    try {
                        const imgRes = await axios.get('/fly/image/' + name);
                        return imgRes.data?.data || '';
                    } catch {
                        return '';
                    }
                })
            );
            setTaskImages(imagesWithData.filter(Boolean));
        } catch (e) {
            message.error('获取图片失败');
        } finally {
            setImageLoading(false);
        }
    };

    const handleMapPointsChange = (points) => {
        setMapPoints(points);
        form.setFieldValue('routeJson', JSON.stringify(points));
    };

    const handleCloseModal = () => {
        setModalOpen(false);
        form.resetFields();
        setMapPoints([]);
    };

    // 统计数据
    const [stats, setStats] = useState({
        total: 0,
        flying: 0,
        charging: 0,
        idle: 0,
    });

    const fetchData = async () => {
        try {
            const [droneRes, taskRes] = await Promise.all([getAllDrones(), getTaskList()]);
            const droneArray = droneRes?.data?.data || droneRes?.data || [];
            const taskArray = taskRes?.data?.data || taskRes?.data || [];

            const droneData = Array.isArray(droneArray) ? droneArray : [];
            const taskData = Array.isArray(taskArray) ? taskArray : [];

            setDrones(droneData);
            setTasks(taskData);

            const newStats = {
                total: droneData.length,
                flying: droneData.filter(d => d && d.status === 'flying').length,
                charging: droneData.filter(d => d && d.status === 'charging').length,
                idle: droneData.filter(d => d && d.status === 'idle').length,
            };
            setStats(newStats);
        } catch (e) {
            console.error('获取数据失败:', e);
            setDrones([]);
            setTasks([]);
            setStats({total: 0, flying: 0, charging: 0, idle: 0});
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 3000);
        return () => clearInterval(interval);
    }, []);

    // 点击无人机卡片查看任务
    useEffect(() => {
        if (droneTasksOpen && selectedDroneNo) {
            getTasksByDrone(selectedDroneNo).then(res => {
                setDroneTasks(res.data?.data || []);
            });
        }
    }, [droneTasksOpen, selectedDroneNo]);

    const handleAddTask = async (values) => {
        try {
            // 转换日期格式
            const data = {...values};
            if (values.startTime && values.startTime[0]) {
                data.startTime = values.startTime[0].toISOString();
            }
            await addTask(data);
            message.success('任务添加成功');
            setModalOpen(false);
            form.resetFields();
            fetchData();
        } catch (e) {
            message.error('添加失败: ' + (e.response?.data?.msg || e.message));
        }
    };

    const handleCancel = async (id) => {
        try {
            await cancelTask(id);
            message.success('任务已取消');
            fetchData();
        } catch (e) {
            message.error('取消失败');
        }
    };

    const handleRetry = async (id) => {
        try {
            await retryTask(id);
            message.success('任务已重试');
            fetchData();
        } catch (e) {
            message.error('重试失败');
        }
    };

    // 重试并设置新开始时间
    const handleRetryWithTime = async (values) => {
        try {
            const id = retryTaskId;
            const data = {...values};
            if (values.startTime && values.startTime[0]) {
                data.startTime = values.startTime[0].toISOString();
            }
            await axios.put('/fly/task/' + id + '/status?status=queued');
            // 如果有时间也更新一下
            if (data.startTime) {
                await axios.put('/fly/task/' + id + '/progress?progress=0');
            }
            message.success('任务已重试');
            setRetryModalOpen(false);
            fetchData();
        } catch (e) {
            message.error('重试失败');
        }
    };

    const openRetryModal = (id) => {
        setRetryTaskId(id);
        setRetryModalOpen(true);
    };

    // 添加无人机
    const handleAddDrone = async (values) => {
        try {
            await addDrone(values);
            message.success('无人机添加成功');
            setDroneModalOpen(false);
            droneForm.resetFields();
            fetchData();
        } catch (e) {
            message.error('添加失败: ' + (e.response?.data?.msg || e.message));
        }
    };

    const columns = [
        {title: 'ID', dataIndex: 'id', key: 'id', width: 60},
        {
            title: '无人机',
            dataIndex: 'droneNo',
            key: 'droneNo',
            width: 90,
            render: (no) => <Tag color="blue">#{no}</Tag>
        },
        {
            title: '状态',
            dataIndex: 'taskStatus',
            key: 'taskStatus',
            width: 90,
            render: (status) => <Tag color={taskStatusColors[status] || 'default'}>{status?.toUpperCase()}</Tag>
        },
        {
            title: '进度',
            dataIndex: 'taskProgress',
            key: 'taskProgress',
            width: 120,
            render: (progress, record) => <Progress percent={Math.round(progress || 0)} size="small"
                                                        status={record.taskStatus === 'failed' ? 'exception' : record.taskStatus === 'finished' ? 'success' : 'active'}
                                                        strokeColor={record.taskStatus === 'running' ? '#52c41a' : undefined}/>
        },
        {
            title: '距离',
            dataIndex: 'dist',
            key: 'dist',
            width: 80,
            render: (dist) => dist ? `${dist.toFixed(2)} km` : '-'
        },
        {
            title: '优先级',
            dataIndex: 'priority',
            key: 'priority',
            width: 70,
            sorter: (a, b) => (b.priority || 0) - (a.priority || 0)
        },
        {
            title: '计划开始',
            dataIndex: 'startTime',
            key: 'startTime',
            width: 150,
            render: (time) => time ? new Date(time).toLocaleString() : '-'
        },
        {
            title: '创建时间',
            dataIndex: 'createTime',
            key: 'createTime',
            width: 150,
            render: (time) => time ? new Date(time).toLocaleString() : '-'
        },
        {
            title: '结果', key: 'result', width: 80, render: (_, record) => {
                if (record.taskStatus === 'finished') {
                    return <Button type="link" size="small" onClick={() => showTaskImages(record.id)}>查看</Button>;
                }
                return '-';
            }
        },
        {
            title: '操作', key: 'action', width: 100, render: (_, record) => (
                <Space>
                    {record.taskStatus === 'queued' || record.taskStatus === 'ready' ? (
                        <Tooltip title="取消"><Button type="text" danger icon={<DeleteOutlined/>}
                                                          onClick={() => handleCancel(record.id)}/></Tooltip>
                    ) : null}
                    {record.taskStatus === 'failed' ? (
                        <Tooltip title="重试">
                            <Button type="text" icon={<SyncOutlined/>} onClick={() => openRetryModal(record.id)}/>
                        </Tooltip>
                    ) : null}
                </Space>
            )
        },
    ];

    // 紧凑模式
    if (compact) {
        return (
            <div>
                <Space style={{width: '100%', justifyContent: 'space-between', marginBottom: 12}}>
                    <Statistic title="飞行中" value={stats.flying} prefix={<SyncOutlined/>}
                               valueStyle={{color: '#1890ff', fontSize: 20}}/>
                    <Statistic title="充电" value={stats.charging} prefix={<ThunderboltOutlined/>}
                               valueStyle={{color: '#52c41a', fontSize: 20}}/>
                    <Statistic title="空闲" value={stats.idle} prefix={<AimOutlined/>}
                               valueStyle={{color: '#8c8c8c', fontSize: 20}}/>
                    <Button type="primary" size="small" icon={<PlusOutlined/>}
                            onClick={() => setModalOpen(true)}>新建</Button>
                </Space>

                <Table
                    columns={columns.filter(c => ['id', 'droneNo', 'taskStatus', 'taskProgress'].includes(c.dataIndex || c.key))}
                    dataSource={tasks} rowKey="id" loading={loading} size="small" pagination={false}
                    scroll={{y: 200}}/>

                <Modal title={<Space><PlusOutlined/>新建任务</Space>} open={modalOpen} onCancel={handleCloseModal}
                       footer={null} width={400}>
                    <Form form={form} layout="vertical" onFinish={handleAddTask}>
                        <Form.Item name="priority" label="优先级" initialValue={5}><InputNumber min={1} max={10}
                                                                                                style={{width: '100%'}}/></Form.Item>
                        <Form.Item name="routeJson" label="飞行路线" rules={[{required: true}]}
                                   extra='点击地图选择航点'>
                            <Input.TextArea rows={3} placeholder='[{"lat":31.23,"lng":121.47},...]'/>
                            <div style={{marginTop: 8}}>
                                <DroneMap onPointsChange={handleMapPointsChange} initialPoints={mapPoints}/>
                            </div>
                        </Form.Item>
                        <Form.Item name="startTime" label="计划时间">
                            <RangePicker showTime={{format: 'HH:mm:ss'}} style={{width: '100%'}}/>
                        </Form.Item>
                        <Form.Item><Space style={{width: '100%', justifyContent: 'flex-end'}}><Button
                            onClick={handleCloseModal}>取消</Button><Button type="primary" htmlType="submit" icon={
                            <SendOutlined/>}>提交</Button></Space></Form.Item>
                    </Form>
                </Modal>
            </div>
        );
    }

    // 完整模式
    return (
        <div style={{padding: '24px'}}>
            <Row gutter={[16, 16]} style={{marginBottom: 24}}>
                <Col xs={12} sm={6}><Card extra={<Button size="small" type="text" icon={<PlusOutlined/>}
                                                             onClick={() => setDroneModalOpen(true)}/>}>
                    <Statistic title="总无人机" value={stats.total} prefix={<RocketOutlined/>}/>
                </Card></Col>
                <Col xs={12} sm={6}><Card style={{borderColor: '#1890ff', borderWidth: 2}}><Statistic title="飞行中"
                                                                                                      value={stats.flying}
                                                                                                      prefix={
                                                                                                          <SyncOutlined
                                                                                                              spin/>}
                                                                                                      valueStyle={{color: '#1890ff'}}/></Card></Col>
                <Col xs={12} sm={6}><Card style={{borderColor: '#52c41a', borderWidth: 2}}><Statistic title="充电中"
                                                                                                      value={stats.charging}
                                                                                                      prefix={
                                                                                                          <ThunderboltOutlined/>}
                                                                                                      valueStyle={{color: '#52c41a'}}/></Card></Col>
                <Col xs={12} sm={6}><Card><Statistic title="空闲" value={stats.idle} prefix={<AimOutlined/>}
                                                     valueStyle={{color: '#8c8c8c'}}/></Card></Col>
            </Row>

            <Row gutter={[16, 16]} style={{marginBottom: 24}}>
                {drones.map(drone => (
                    <Col xs={24} sm={12} md={8} lg={6} key={drone.droneNo}>
                        <Card
                            size="small"
                            hoverable
                            clickable
                            onClick={() => {
                                setSelectedDroneNo(drone.droneNo);
                                setDroneTasksOpen(true);
                            }}
                            style={{
                                borderColor: drone.status === 'flying' ? '#1890ff' : drone.status === 'charging' ? '#52c41a' : drone.status === 'error' ? '#ff4d4f' : '#d9d9d9',
                                borderWidth: 2,
                                cursor: 'pointer'
                            }}
                            title={<Space><RocketOutlined/>无人机 #{drone.droneNo}</Space>}
                            extra={<Badge status={statusColors[drone.status] || 'default'}
                                          text={drone.status?.toUpperCase()}/>}
                        >
                            <Row gutter={8}>
                                <Col span={12}><Statistic title="电量" value={Number(drone.power?.toFixed(1)) || 0}
                                                          suffix="%" valueStyle={{
                                    fontSize: 18,
                                    color: drone.power < 20 ? '#ff4d4f' : '#52c41a'
                                }}/></Col>
                                <Col span={12}><Statistic title="高度" value={drone.altitude || 0} suffix="m"
                                                          valueStyle={{fontSize: 18}}/></Col>
                                <Col span={12}><Statistic title="速度" value={Number(drone.speed?.toFixed(1)) || 0}
                                                          suffix="m/s" valueStyle={{fontSize: 18}}/></Col>
                                <Col span={12}><Statistic title="GPS" value={Math.round(drone.gpsSignal || 0)}
                                                          suffix="%" valueStyle={{fontSize: 18}}/></Col>
                            </Row>
                        </Card>
                    </Col>
                ))}
            </Row>

            <Card title={<Space><RocketOutlined/>任务列表<Tag>{tasks.length}</Tag></Space>}
                  extra={<Button type="primary" icon={<PlusOutlined/>}
                                 onClick={() => setModalOpen(true)}>新建任务</Button>}>
                <Table columns={columns} dataSource={tasks} rowKey="id" loading={loading}
                       pagination={{pageSize: 10}} scroll={{x: 900}}/>
            </Card>

            <Modal title={<Space><PlusOutlined/>新建飞行任务</Space>} open={modalOpen} onCancel={handleCloseModal}
                   footer={null} width={500}>
                <Form form={form} layout="vertical" onFinish={handleAddTask}>
                    <Form.Item name="priority" label="优先级" initialValue={5}><InputNumber min={1} max={10}
                                                                                            style={{width: '100%'}}/></Form.Item>
                    <Form.Item name="routeJson" label="飞行路线 (JSON)" rules={[{required: true}]}
                               extra='点击地图选择航点'>
                        <Input.TextArea rows={4}
                                        placeholder='[{"lat": 31.2304, "lng": 121.4737}, {"lat": 31.2310, "lng": 121.4742}]'/>
                        <div style={{marginTop: 8}}>
                            <DroneMap onPointsChange={handleMapPointsChange} initialPoints={mapPoints}/>
                        </div>
                    </Form.Item>
                    <Form.Item name="startTime" label="计划开始时间">
                        <RangePicker showTime={{format: 'HH:mm:ss'}} style={{width: '100%'}}/>
                    </Form.Item>
                    <Form.Item><Space style={{width: '100%', justifyContent: 'flex-end'}}><Button
                        onClick={handleCloseModal}>取消</Button><Button type="primary" htmlType="submit" icon={
                        <SendOutlined/>}>提交任务</Button></Space></Form.Item>
                </Form>
            </Modal>

            {/* 重试弹窗 */}
            <Modal title={<Space><PlusOutlined/>重试任务</Space>} open={retryModalOpen}
                   onCancel={() => setRetryModalOpen(false)} footer={null} width={400}>
                <Form form={retryForm} layout="vertical" onFinish={handleRetryWithTime}>
                    <Form.Item name="startTime" label="新计划开始时间">
                        <RangePicker showTime={{format: 'HH:mm:ss'}} style={{width: '100%'}}/>
                    </Form.Item>
                    <Form.Item><Space style={{width: '100%', justifyContent: 'flex-end'}}><Button
                        onClick={() => setRetryModalOpen(false)}>取消</Button><Button type="primary"
                                                                                      htmlType="submit" icon={
                        <SendOutlined/>}>确认重试</Button></Space></Form.Item>
                </Form>
            </Modal>

            {/* 添加无人机弹窗 */}
            <Modal title={<Space><RocketOutlined/>添加无人机</Space>} open={droneModalOpen}
                   onCancel={() => setDroneModalOpen(false)} footer={null} width={400}>
                <Form form={droneForm} layout="vertical" onFinish={handleAddDrone}>
                    <Form.Item name="droneNo" label="无人机编号" rules={[{required: true}]}>
                        <InputNumber min={1} style={{width: '100%'}}/>
                    </Form.Item>
                    <Form.Item name="power" label="初始电量" initialValue={100}>
                        <InputNumber min={0} max={100} style={{width: '100%'}}/>
                    </Form.Item>
                    <Form.Item><Space style={{width: '100%', justifyContent: 'flex-end'}}><Button
                        onClick={() => setDroneModalOpen(false)}>取消</Button><Button type="primary"
                                                                                      htmlType="submit" icon={
                        <SendOutlined/>}>添加</Button></Space></Form.Item>
                </Form>
            </Modal>

            {/* 无人机任务详情弹窗 */}
            <Modal
                title={<Space><RocketOutlined/> 无人机 #{selectedDroneNo} 任务队列</Space>}
                open={droneTasksOpen}
                onCancel={() => setDroneTasksOpen(false)}
                footer={null}
                width={600}
            >
                <Table
                    dataSource={droneTasks}
                    rowKey="id"
                    size="small"
                    pagination={false}
                    columns={[
                        {title: 'ID', dataIndex: 'id', width: 60},
                        {
                            title: '状态',
                            dataIndex: 'taskStatus',
                            render: (s) => <Tag color={taskStatusColors[s]}>{s}</Tag>
                        },
                        {title: '进度', dataIndex: 'taskProgress', render: (p) => p ? Math.round(p) + '%' : '-'},
                        {
                            title: '创建',
                            dataIndex: 'createTime',
                            render: (t) => t ? new Date(t).toLocaleDateString() : '-'
                        },
                    ]}
                />
            </Modal>

            {/* 任务结果图片弹窗 */}
            <Modal
                title="任务结果图片"
                open={imageModalOpen}
                onCancel={() => setImageModalOpen(false)}
                footer={null}
                width={800}
                loading={imageLoading}
            >
                <Row gutter={[8, 8]}>
                    {taskImages.map((src, idx) => (
                        <Col xs={12} sm={8} key={idx}>
                            <img src={src} alt={`result_${idx}`} style={{width: '100%', borderRadius: 4}}/>
                        </Col>
                    ))}
                </Row>
                {taskImages.length === 0 && !imageLoading && (
                    <Empty description="暂无图片"/>
                )}
            </Modal>
        </div>
    );
};

export default DroneTask;
