# Body Measurement API - Frontend

A modern Next.js dashboard for the Body Measurement API platform. This B2B SaaS frontend enables e-commerce brands to measure body dimensions from photos and provide accurate clothing size recommendations.

## Features

- **Authentication**: Secure brand registration and login
- **Dashboard**: Real-time statistics and usage metrics
- **Image Upload**: Process body measurements from photos
- **Product Management**: Add products with custom size charts
- **Analytics**: Track performance, ROI, and size distributions
- **API Integration**: Full integration with FastAPI backend

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **State Management**: React Hooks

## Project Structure

```
body-measurement-frontend/
├── app/
│   ├── auth/
│   │   ├── login/page.tsx          # Login page
│   │   └── register/page.tsx       # Registration page
│   ├── dashboard/
│   │   ├── layout.tsx              # Dashboard layout with navigation
│   │   ├── page.tsx                # Dashboard home (stats)
│   │   ├── upload/page.tsx         # Image upload & results
│   │   ├── products/page.tsx       # Product management
│   │   ├── analytics/page.tsx      # Analytics dashboard
│   │   └── api-keys/page.tsx       # API key management
│   ├── layout.tsx                  # Root layout
│   ├── page.tsx                    # Landing page
│   └── globals.css                 # Global styles
├── lib/
│   ├── api.ts                      # API client functions
│   ├── auth.ts                     # Auth helpers
│   └── types.ts                    # TypeScript types
└── package.json
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

### Installation

1. Install dependencies:
```bash
npm install
```

2. Create `.env.local` file (already created):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

### 1. Register an Account

- Click "Get Started" or "Sign Up"
- Enter your brand name, email, and password
- You'll be automatically logged in after registration

### 2. Upload Images

- Navigate to "Upload Image"
- Select a full-body photo
- Click "Process Measurements"
- View detailed measurements and size recommendations

### 3. Manage Products

- Go to "Products"
- Click "Add Product"
- Enter product details and size chart
- Save to use for size matching

### 4. View Analytics

- Visit "Analytics" to see:
  - Total measurements
  - Size distribution
  - ROI metrics
  - Popular products

### 5. API Integration

- Go to "API Keys"
- Copy your API key
- Use the provided cURL examples to integrate with your platform

## API Endpoints Used

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/auth/register` | POST | Register new brand |
| `/api/v1/auth/login` | POST | Login and get JWT |
| `/api/v1/brands/me` | GET | Get brand profile |
| `/api/v1/brands/usage` | GET | Get usage statistics |
| `/api/v1/brands/analytics` | GET | Get analytics data |
| `/api/v1/brands/products` | GET/POST | Manage products |
| `/api/v1/measurements/process` | POST | Process image measurements |

## Available Scripts

```bash
# Development server
npm run dev

# Production build
npm run build

# Start production server
npm start

# Run linter
npm run lint
```

## Key Components

### Authentication Flow

1. User registers/logs in
2. Backend returns JWT token and API key
3. Frontend stores in localStorage
4. Token included in subsequent requests

### Image Processing

1. User uploads image
2. Frontend sends to backend via FormData
3. Backend processes with MediaPipe/OpenCV
4. Results displayed with confidence scores

### Product Management

- Add products with custom size charts
- Define measurements for each size (XS-XL)
- Used for matching measurements to sizes

## Styling

- Uses Tailwind CSS utility classes
- Responsive design (mobile + desktop)
- Custom gradient backgrounds
- Consistent color scheme (Indigo primary)

## Security

- JWT tokens stored in localStorage
- API keys never exposed in client code
- Environment variables for API URL
- Input validation on all forms

## Integration with Backend

### CORS Setup Required

Make sure your backend `.env` includes:

```env
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

### Testing Connection

1. Start backend: `docker-compose up` (in backend directory)
2. Start frontend: `npm run dev` (in this directory)
3. Register a test account
4. Upload a test image

## Troubleshooting

### "API key not found" Error
- Clear localStorage and log in again
- Check backend is running on port 8000

### CORS Errors
- Update backend CORS_ORIGINS in `.env`
- Restart backend after changing

### Image Upload Fails
- Check file size (max 10MB)
- Ensure image format is JPG/PNG/WEBP
- Verify backend MediaPipe setup

## Future Enhancements

- [ ] Dark mode toggle
- [ ] Batch image processing
- [ ] Export results as PDF
- [ ] Advanced filtering in analytics
- [ ] Webhook configuration
- [ ] Multi-language support
- [ ] Real-time notifications

## Contributing

This is a B2B SaaS platform. For feature requests or bug reports, please contact the development team.

## License

Proprietary - All rights reserved

## Contact

For support or questions about the Body Measurement API platform, please reach out to your account manager.
