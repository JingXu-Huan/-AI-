import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { spawn } from 'node:child_process'
import formidable from 'formidable'

function pythonDetectionPlugin() {
  return {
    name: 'python-detection-plugin',
    configureServer(server) {
      // 新增：获取frames目录下的帧列表
      server.middlewares.use('/api/frames', (req, res, next) => {
        try {
          const urlObj = new URL(req.url, 'http://localhost');
          const stem = urlObj.searchParams.get('stem');
          if (stem) {
            const projectRoot = path.resolve(fs.realpathSync('.'), '..');
            const framesDir = path.join(projectRoot, 'outputs', stem, 'frames');
            if (fs.existsSync(framesDir) && fs.statSync(framesDir).isDirectory()) {
              const files = fs.readdirSync(framesDir).filter(f => f.endsWith('.jpg')).sort();
              res.writeHead(200, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ success: true, frames: files }));
              return;
            }
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ success: true, frames: [] }));
        } catch(e) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: e.message }));
        }
      });

      // 暴露一个新路由用来读取本地的图片/视频/帧
      server.middlewares.use('/api/image', (req, res, next) => {
        try {
          const urlObj = new URL(req.url, 'http://localhost');
          const stem = urlObj.searchParams.get('stem');
          if (stem) {
             const projectRoot = path.resolve(fs.realpathSync('.'), '..');
             
             // 1. 优先检查 frames 目录（有标记的帧）
             const framesDir = path.join(projectRoot, 'outputs', stem, 'frames');
             if (fs.existsSync(framesDir) && fs.statSync(framesDir).isDirectory()) {
                const frameIndex = urlObj.searchParams.get('frame');
                let framePath;
                if (frameIndex) {
                   framePath = path.join(framesDir, 'frame_' + frameIndex.padStart(5, '0') + '.jpg');
                } else {
                   const files = fs.readdirSync(framesDir).filter(f => f.endsWith('.jpg')).sort();
                   if (files.length > 0) {
                      framePath = path.join(framesDir, files[0]);
                   }
                }
                if (framePath && fs.existsSync(framePath)) {
                   res.setHeader('Content-Type', 'image/jpeg');
                   fs.createReadStream(framePath).pipe(res);
                   return;
                }
             }
             
             // 2. 检查普通标注图片
             const imgPath = path.join(projectRoot, 'outputs', stem, stem + '_annotated.jpg');
             if (fs.existsSync(imgPath)) {
               res.setHeader('Content-Type', 'image/jpeg');
               fs.createReadStream(imgPath).pipe(res);
               return;
             }
             
             // 3. 检查视频 (AVI 或 MP4)
             let videoPath = path.join(projectRoot, 'outputs', stem, stem + '_annotated.mp4');
             let videoType = 'video/mp4';
             if (!fs.existsSync(videoPath)) {
               videoPath = path.join(projectRoot, 'outputs', stem, stem + '_annotated.avi');
               videoType = 'video/x-msvideo';
             }
             
             if (fs.existsSync(videoPath)) {
               const stat = fs.statSync(videoPath);
               const fileSize = stat.size;
               const range = req.headers.range;

               if (range) {
                 const parts = range.replace(/bytes=/, "").split("-");
                 const start = parseInt(parts[0], 10);
                 const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
                 if (start >= fileSize) {
                   res.writeHead(416);
                   return res.end();
                 }
                 const chunksize = (end - start) + 1;
                 const fileStream = fs.createReadStream(videoPath, { start, end });
                 res.writeHead(206, {
                   'Content-Range': `bytes ${start}-${end}/${fileSize}`,
                   'Accept-Ranges': 'bytes',
                   'Content-Length': chunksize,
                   'Content-Type': videoType,
                 });
                 fileStream.pipe(res);
               } else {
                 res.writeHead(200, {
                   'Content-Type': videoType,
                 });
                 fs.createReadStream(videoPath).pipe(res);
               }
               return;
             }
          }
        } catch(e) { console.error('API image error:', e); }
        next();
      });

      let liveProcess = null;

      server.middlewares.use('/api/stream_live', (req, res, next) => {
        try {
          const urlObj = new URL(req.url, 'http://localhost');
          const source = urlObj.searchParams.get('source') || '0';
          const location = urlObj.searchParams.get('location') || '大门主干道';
          
          if (liveProcess) {
             liveProcess.kill();
             liveProcess = null;
          }

          res.writeHead(200, {
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          });

          const projectRoot = path.resolve(fs.realpathSync('.'), '..');
          const pythonExe = process.platform === 'win32' 
            ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
            : path.join(projectRoot, '.venv', 'bin', 'python');
          
          liveProcess = spawn(pythonExe, ['main.py', source, '--location', location, '--mjpeg'], { cwd: projectRoot });
          liveProcess.stdout.pipe(res);
          
          req.on('close', () => {
             if (liveProcess) {
                liveProcess.kill();
                liveProcess = null;
             }
          });
          return;
        } catch(e) {}
        next();
      });

      server.middlewares.use('/api/stream_stop', (req, res, next) => {
        if (liveProcess) {
           liveProcess.kill();
           liveProcess = null;
        }
        try {
           const projectRoot = path.resolve(fs.realpathSync('.'), '..');
           const jsonPath = path.join(projectRoot, 'outputs', 'stream', 'stream.json');
           let detectionResult = [];
           if (fs.existsSync(jsonPath)) {
               const fileContent = fs.readFileSync(jsonPath, 'utf-8');
               detectionResult = JSON.parse(fileContent);
           }
           res.writeHead(200, { 'Content-Type': 'application/json' });
           res.end(JSON.stringify({ success: true, data: detectionResult, stem: 'stream' }));
        } catch(e) {
           res.writeHead(500, { 'Content-Type': 'application/json' });
           res.end(JSON.stringify({ error: e.message }));
        }
      });

      server.middlewares.use('/api/detect', (req, res, next) => {
        if (req.method === 'POST') {
          const form = formidable({ multiples: false, keepExtensions: true, maxFileSize: 2 * 1024 * 1024 * 1024 });
          form.parse(req, (err, fields, files) => {
            if (err) {
              res.writeHead(500, { 'Content-Type': 'application/json' });
              return res.end(JSON.stringify({ error: err.message }));
            }
            const fileArray = files.file;
            let file = undefined;
            if (fileArray) {
               file = Array.isArray(fileArray) ? fileArray[0] : fileArray;
            }
            
            let streamUrl = undefined;
            if (fields.stream_url) {
               streamUrl = Array.isArray(fields.stream_url) ? fields.stream_url[0] : fields.stream_url;
            }

            if (!file && !streamUrl) {
              res.writeHead(400, { 'Content-Type': 'application/json' });
              return res.end(JSON.stringify({ error: 'No file or stream URL uploaded' }));
            }

            const projectRoot = path.resolve(fs.realpathSync('.'), '..');
            let targetPath = null;
            let originalName = null;
            
            if (file) {
                originalName = file.originalFilename || 'upload.bin';
                targetPath = path.join(projectRoot, originalName);
                fs.copyFileSync(file.filepath, targetPath);
            }

            let location = 'A区-3号楼';
            if (fields.location) {
               location = Array.isArray(fields.location) ? fields.location[0] : fields.location;
            }
            
            let maxFramesArgs = [];
            if (fields.max_frames) {
               let mf = Array.isArray(fields.max_frames) ? fields.max_frames[0] : fields.max_frames;
               maxFramesArgs = ['--max-frames', mf];
            }
            
            const pythonExe = process.platform === 'win32' 
              ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
              : path.join(projectRoot, '.venv', 'bin', 'python');
            const sourceArg = streamUrl ? streamUrl : targetPath;

            const child = spawn(pythonExe, ['main.py', sourceArg, '--location', location, '--frame-interval', '5', ...maxFramesArgs], { cwd: projectRoot });
            let stderrData = '';

            child.stdout.on('data', (data) => {});
            child.stderr.on('data', (data) => {
               stderrData += data.toString();
            });

            child.on('error', (execErr) => {
               if (targetPath) { try { fs.unlinkSync(targetPath); } catch(e) {} }
               res.writeHead(500, { 'Content-Type': 'application/json' });
               return res.end(JSON.stringify({ error: 'Detection failed', details: execErr.message }));
            });

            child.on('close', (code) => {
              if (targetPath) { try { fs.unlinkSync(targetPath); } catch(e) {} }
              if (code !== 0) {
                 res.writeHead(500, { 'Content-Type': 'application/json' });
                 return res.end(JSON.stringify({ error: 'Detection failed', details: stderrData }));
              }
              try {
                 let stem = streamUrl ? (streamUrl === '0' || /^\d+$/.test(streamUrl) ? 'camera_' + streamUrl : 'stream') : path.parse(originalName).name;
                 const jsonPath = path.join(projectRoot, 'outputs', stem, stem + '.json');
                 let detectionResult = [];
                 if (fs.existsSync(jsonPath)) {
                     const fileContent = fs.readFileSync(jsonPath, 'utf-8');
                     detectionResult = JSON.parse(fileContent);
                 }
                 res.writeHead(200, { 'Content-Type': 'application/json' });
                 res.end(JSON.stringify({ success: true, data: detectionResult, stem: stem }));
              } catch (e) {
                 res.writeHead(500, { 'Content-Type': 'application/json' });
                 res.end(JSON.stringify({ error: 'Failed to access results', details: e.message }));
              }
            });
          });
          return;
        }
        next();
      });
    }
  }
}

export default defineConfig({
  server: {
    proxy: {
      '^/api/(getDesc|writeToDB|repairs).*': {
        target: 'http://localhost:8080',
        changeOrigin: true
      }
    }
  },
  plugins: [react(), pythonDetectionPlugin()],
})
