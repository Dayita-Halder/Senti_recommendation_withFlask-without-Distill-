# Deploying to Vercel

This guide explains how to deploy the Sentiment-Based Recommendation System Flask app to Vercel.

## **Option 1: Deploy via Vercel Dashboard (Easiest)**

### Step 1: Connect GitHub Repository
1. Go to [Vercel.com](https://vercel.com)
2. Sign in with GitHub (or create an account)
3. Click **"Add New..."** → **"Project"**
4. Select your repository: `Senti_recommendation_withFlask-without-Distill-`
5. Click **"Import"**

### Step 2: Configure Project
1. **Framework Preset:** Select **"Other"** (or **"Python"**)
2. **Root Directory:** Leave as default (`.`)
3. **Build Command:** Leave empty (Vercel will auto-detect)
4. **Output Directory:** Leave empty
5. Click **"Deploy"**

### Step 3: Wait for Deployment
- Vercel will build and deploy your app automatically
- You'll get a URL like: `https://senti-recommendation-xxxxx.vercel.app`

---

## **Option 2: Deploy via Vercel CLI**

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Deploy from Your Project Directory
```bash
cd d:\Senti-recommen2
vercel
```

### Step 3: Follow the Prompts
```
? Set up and deploy "d:\Senti-recommen2"? [Y/n] y
? Which scope do you want to deploy to? (select your account)
? Link to existing project? [y/N] n
? What's your project's name? senti-recommendation-2
? In which directory is your code located? ./
? Want to modify these settings? [y/N] n
```

Your app will be deployed at the provided URL.

---

## **Issues & Solutions**

### Issue: "Module not found" errors
**Solution:** Ensure `api/index.py` and `vercel.json` are in the correct locations

### Issue: Large pickle files exceeding limits
**Solution:** 
- Vercel has a limit of 500MB per deployment
- If pickle files are too large, consider:
  - Using a database instead of pickle files
  - Splitting pickle files into smaller chunks
  - Using a cloud storage service (AWS S3, Google Cloud Storage)

### Issue: StatelessRequest/WSGI conversion errors
**Solution:** This is already handled in `api/index.py` with proper WSGI wrapping

---

## **Environment Variables (Optional)**

If you need environment variables on Vercel:

1. Go to your project dashboard on Vercel
2. Click **"Settings"** → **"Environment Variables"**
3. Add variables like:
   - `FLASK_ENV=production`
   - `FLASK_DEBUG=0`

---

## **Monitoring & Logs**

After deployment:
1. Go to your Vercel dashboard
2. Click on your project
3. View **"Deployments"** to see logs
4. Click on active deployment to see real-time logs

---

## **Custom Domain (Optional)**

1. In Vercel dashboard, go to **"Settings"** → **"Domains"**
2. Add your custom domain
3. Follow the DNS configuration instructions

---

## **API Endpoints on Vercel**

Once deployed, your API will be available at:

```
https://your-project.vercel.app/
https://your-project.vercel.app/api/recommend
https://your-project.vercel.app/api/sentiment
https://your-project.vercel.app/api/users
https://your-project.vercel.app/api/products
https://your-project.vercel.app/health
```

---

## **LocalStorage Considerations**

**Important:** Vercel serverless functions don't have persistent storage between requests. The pickle files are loaded at startup, which works fine. However:

- ✅ Reading pickle files works
- ✅ Making predictions works
- ❌ Writing new data to disk won't persist between deployments

For production use with data persistence, consider:
1. **AWS RDS** for database
2. **Firebase** for real-time data
3. **Supabase** for PostgreSQL hosting
4. **MongoDB Atlas** for document storage

---

## **Troubleshooting**

### Check Deployment Status
```bash
vercel deployments
```

### View Live Logs
```bash
vercel logs [project-url]
```

### Rollback to Previous Deployment
```bash
vercel rollback [project-name]
```

---

## **Next Steps**

1. **Push the new files to GitHub:**
   ```bash
   git add vercel.json api/ .vercelignore requirements.txt
   git commit -m "Add Vercel deployment configuration"
   git push origin main
   ```

2. **Connect to Vercel** using the dashboard or CLI

3. **Test the deployed API** using a tool like Postman or cURL

4. **Monitor performance** in the Vercel dashboard

---

**Your Flask app is now ready for Vercel deployment!** 🚀
