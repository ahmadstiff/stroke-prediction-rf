# 🚀 Deployment Guide - Stroke Prediction App

Panduan lengkap untuk deploy API ke Render dan Frontend ke Vercel.

## 📁 Struktur Proyek

```
stroke-prediction-rf/
├── api/                    # Backend API (Render)
│   ├── main.py            # FastAPI application
│   ├── requirements.txt   # Python dependencies
│   ├── render.yaml        # Render configuration
│   ├── models/            # ML models
│   └── test_api.py        # API testing script
├── frontend/              # Frontend (Vercel)
│   ├── app/               # Next.js app directory
│   ├── package.json       # Node.js dependencies
│   ├── vercel.json        # Vercel configuration
│   └── tailwind.config.js # Tailwind CSS config
└── data/                  # Dataset
```

## 🔧 Deploy API ke Render

### 1. Persiapan Repository

1. **Push ke GitHub:**
```bash
git add .
git commit -m "Add API and Frontend for deployment"
git push origin main
```

2. **Pastikan struktur API folder:**
```
api/
├── main.py
├── requirements.txt
├── render.yaml
├── models/
│   ├── random_forest_model_97.74%.pkl
│   ├── scaler_97.74%.pkl
│   ├── encoder_97.74%.pkl
│   └── feature_selector_97.74%.pkl
└── test_api.py
```

### 2. Deploy di Render

1. **Buka [render.com](https://render.com)**
2. **Sign up/Login dengan GitHub**
3. **Klik "New +" → "Web Service"**
4. **Connect repository GitHub**
5. **Konfigurasi deployment:**

```
Name: stroke-prediction-api
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

6. **Klik "Create Web Service"**
7. **Tunggu deployment selesai (5-10 menit)**

### 3. Verifikasi API

1. **Test health check:**
```bash
curl https://your-api-name.onrender.com/
```

2. **Test prediction:**
```bash
curl -X POST https://your-api-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 120.5,
    "weight": 70.0,
    "height": 170.0,
    "smoking_status": "never smoked"
  }'
```

## 🎨 Deploy Frontend ke Vercel

### 1. Persiapan Frontend

1. **Masuk ke folder frontend:**
```bash
cd frontend
```

2. **Install dependencies:**
```bash
npm install
```

3. **Test local:**
```bash
npm run dev
```

### 2. Deploy di Vercel

1. **Buka [vercel.com](https://vercel.com)**
2. **Sign up/Login dengan GitHub**
3. **Klik "New Project"**
4. **Import repository GitHub**
5. **Konfigurasi project:**

```
Framework Preset: Next.js
Root Directory: frontend
Build Command: npm run build
Output Directory: .next
Install Command: npm install
```

6. **Environment Variables:**
```
NEXT_PUBLIC_API_URL=https://your-api-name.onrender.com
```

7. **Klik "Deploy"**
8. **Tunggu deployment selesai (2-3 menit)**

### 3. Verifikasi Frontend

1. **Buka URL Vercel yang diberikan**
2. **Test form input dengan data sample**
3. **Verifikasi response dari API**

## 🔍 Testing

### Test API Lokal

```bash
cd api
python test_api.py
```

### Test Frontend Lokal

```bash
cd frontend
npm run dev
# Buka http://localhost:3000
```

## 📊 Monitoring

### Render Dashboard
- **Logs:** Monitor API logs di Render dashboard
- **Metrics:** CPU, memory usage
- **Health:** Auto-restart jika crash

### Vercel Dashboard
- **Analytics:** Page views, performance
- **Functions:** Serverless function logs
- **Deployments:** Build status, rollback

## 🔧 Troubleshooting

### API Issues

1. **Model not loaded:**
   - Pastikan file model ada di folder `models/`
   - Check file permissions

2. **Import errors:**
   - Update `requirements.txt` dengan versi yang benar
   - Rebuild di Render

3. **CORS errors:**
   - Update CORS settings di `main.py`
   - Add frontend domain ke allowed origins

### Frontend Issues

1. **Build errors:**
   - Check TypeScript errors
   - Update dependencies

2. **API connection:**
   - Verify `NEXT_PUBLIC_API_URL` environment variable
   - Test API endpoint manually

3. **Styling issues:**
   - Check Tailwind CSS configuration
   - Verify CSS imports

## 🚀 Production Checklist

### API (Render)
- [ ] Model files uploaded to `models/` folder
- [ ] Requirements.txt updated
- [ ] CORS configured for frontend domain
- [ ] Health check endpoint working
- [ ] Error handling implemented
- [ ] API tested with sample data

### Frontend (Vercel)
- [ ] Environment variables set
- [ ] API URL configured
- [ ] Form validation working
- [ ] Error handling implemented
- [ ] Responsive design tested
- [ ] Loading states implemented

## 📈 Performance Optimization

### API
- **Caching:** Implement Redis for model caching
- **Compression:** Enable gzip compression
- **CDN:** Use Cloudflare for global distribution

### Frontend
- **Image optimization:** Next.js automatic optimization
- **Code splitting:** Automatic with Next.js
- **Caching:** Static generation where possible

## 🔒 Security

### API Security
- **Rate limiting:** Implement request limits
- **Input validation:** Strict validation for all inputs
- **HTTPS:** Automatic with Render
- **CORS:** Restrict to specific domains

### Frontend Security
- **Environment variables:** Never expose API keys
- **Input sanitization:** Validate all user inputs
- **HTTPS:** Automatic with Vercel

## 📞 Support

### Render Support
- **Documentation:** [render.com/docs](https://render.com/docs)
- **Community:** [render.com/community](https://render.com/community)

### Vercel Support
- **Documentation:** [vercel.com/docs](https://vercel.com/docs)
- **Community:** [github.com/vercel/vercel/discussions](https://github.com/vercel/vercel/discussions)

---

## 🎯 Quick Deploy Commands

### API (Render)
```bash
# 1. Push to GitHub
git add .
git commit -m "Deploy API"
git push origin main

# 2. Deploy on Render (via web interface)
# 3. Get API URL and update frontend
```

### Frontend (Vercel)
```bash
# 1. Set environment variable
# NEXT_PUBLIC_API_URL=https://your-api.onrender.com

# 2. Deploy on Vercel (via web interface)
# 3. Test the application
```

**🎉 Selamat! Aplikasi Anda sudah siap digunakan!** 