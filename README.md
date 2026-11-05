# 🐍 Venom Watch
### AI-Powered Snake Incident Monitoring and Public Safety Platform

**Venom Watch** is an AI-powered snake incident monitoring and public safety platform designed to improve snakebite awareness, incident reporting, and emergency response.

The platform enables citizens to report snake sightings, analyze snake images using machine learning, identify potential venomous species, and monitor snake-related incidents through geographical visualization.

It also provides authorities with tools to monitor reported incidents, identify geographical hotspots, manage emergency responses, and coordinate responders.

The primary objective of Venom Watch is to bridge the gap between snake incident reporting, AI-assisted identification, and timely emergency response.

---

## 🎯 Problem Statement

Snake encounters and snakebite incidents can pose serious public safety challenges, particularly in regions where people frequently encounter venomous snakes.

Traditional reporting methods often lack centralized monitoring, geographical analysis, and efficient coordination between citizens and emergency response authorities.

Venom Watch addresses these challenges through an integrated platform combining artificial intelligence, machine learning, geographical information systems (GIS), and centralized incident management.

## ✨ Key Features

### 1. 🐍 AI-Based Snake Identification
- Upload snake images for AI-powered analysis.
- CNN-based image classification to identify snake species.
- Classification across 15 snake species and a NOT_SNAKE category.
- Identify whether the predicted species is venomous or non-venomous.
- Display prediction confidence to assist users.
- Reject images classified as non-snake.

### 2. 📍 Citizen Incident Reporting
- Submit snake sighting reports.
- Upload supporting snake images.
- Record incident locations.
- Maintain structured incident information.
- Support centralized monitoring of reported sightings.

### 3. 🗺️ GIS-Based Hotspot Detection
- Visualize reported snake incidents geographically.
- Apply DBSCAN (Density-Based Spatial Clustering of Applications with Noise) for spatial clustering.
- Identify areas with concentrated snake sightings.
- Detect geographical hotspots to support public safety planning.

### 4. 🚨 Authority Monitoring Dashboard
- Monitor reported snake incidents.
- View incident information and geographical distribution.
- Track reported cases.
- Support incident management and response coordination.

### 5. 👨‍🚒 Responder Management
- Assign responders to reported incidents.
- Monitor response assignments.
- Support coordination between authorities and emergency responders.

### 6. 🩹 Wound Screening
- Provide a wound-screening module to assist with preliminary assessment.
- Support the collection and evaluation of wound-related information.

*Wound screening is intended as an assistance feature, not a substitute for professional medical assessment.*

### 7. 🔍 Duplicate Report Detection
- Identify potentially duplicate snake sighting reports.
- Reduce redundant incident records.
- Improve the consistency of incident monitoring.

---

## 🧠 AI & Machine Learning Implementation

Venom Watch integrates machine learning into its snake identification and geographical analysis modules.

### Snake Species Classification

A CNN-based image classification approach is used to identify snake species from uploaded images.

**Classification categories:**
- 15 snake species.
- NOT_SNAKE category for non-snake images.

**Model implementation:**
- Convolutional Neural Network (CNN).
- ResNet18 transfer learning.
- Image preprocessing and normalization.
- Multi-class classification.
- Confidence score generation.

The classification pipeline predicts the snake species and determines its venomous or non-venomous category using the species mapping.

### Spatial Hotspot Detection

The DBSCAN clustering algorithm is used to identify geographical concentrations of snake sightings.

**Why DBSCAN?**
- Does not require a predefined number of clusters.
- Identifies dense geographical regions.
- Detects noise and isolated observations.
- Suitable for geographical incident analysis.

### AI Processing Workflow

1. Citizen uploads a snake image.
2. Image preprocessing is performed.
3. CNN model analyzes the image.
4. The predicted species and confidence score are generated.
5. Species mapping determines the venom classification.
6. The system processes the result for the reporting workflow.

**Important:** AI predictions are intended to support identification and reporting, not to guarantee species identification or determine whether an area is safe.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Backend | Python, Django |
| Database | SQLite |
| Machine Learning | PyTorch, CNN, ResNet18 |
| Computer Vision | Image preprocessing |
| Data Analysis | NumPy, Pandas |
| Spatial Clustering | DBSCAN, Scikit-learn |
| Frontend | HTML, CSS, JavaScript |
| GIS | Geographical coordinates and spatial visualization |
| Development Tools | VS Code, Git, GitHub |

---

## 🏗️ System Architecture

Venom Watch follows a modular architecture that integrates the web application, AI processing, geographical analysis, and incident management.

**Major components:**

1. **Presentation Layer:** Citizen and authority interfaces.
2. **Application Layer:** Django backend and business logic.
3. **AI Layer:** Snake image classification and prediction.
4. **GIS Layer:** Spatial clustering and hotspot identification.
5. **Database Layer:** Incident records, users, reports, and related information.

---

## 📂 Project Modules

| Module | Responsibility |
|---|---|
| User Management | User registration and authentication |
| Snake Detection | Image analysis and snake identification |
| Citizen Reporting | Snake sighting submissions |
| Duplicate Detection | Identification of repeated reports |
| GIS Analysis | Geographical visualization and DBSCAN clustering |
| Authority Dashboard | Incident monitoring and management |
| Responder Management | Responder assignment and coordination |
| Wound Screening | Preliminary wound assessment |

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or compatible version
- Git
- pip
- Virtual environment

### 1. Clone the Repository

```bash
git clone https://github.com/Nidhish-Chandran/Venom-Watch.git
```

### 2. Navigate to the Project Directory

```bash
cd Venom-Watch
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

If a requirements file is available:

```bash
pip install -r requirements.txt
```

### 6. Apply Database Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 7. Create a Superuser

```bash
python manage.py createsuperuser
```

### 8. Run the Development Server

```bash
python manage.py runserver
```

Open the application in your browser:

```text
http://127.0.0.1:8000/
```

**Note:** AI model weights, environment variables, API credentials, and other required assets must be configured separately according to the project setup.

---

## 🔬 Machine Learning Algorithms

| Algorithm / Model | Application |
|---|---|
| CNN | Snake species classification |
| ResNet18 | Transfer learning for image classification |
| DBSCAN | Geographical hotspot detection |

---

## 🚀 Future Enhancements

- Improve snake classification accuracy with larger and more diverse datasets.
- Expand the supported snake species.
- Enhance real-time geographical incident monitoring.
- Integrate emergency notification services.
- Improve responder tracking and coordination.
- Enhance AI-assisted wound assessment.
- Migrate the database to PostgreSQL for improved scalability.
- Develop a dedicated mobile application.

---

## 👨‍💻 Project Information

| Detail | Information |
|---|---|
| Project Name | Venom Watch |
| Degree | Master of Computer Applications (MCA) |
| Institution | TKM College of Engineering, Kollam |
| Department | Department of Computer Applications |
| Academic Year | 2025–2027 |
| Project Guide | Natheera Beevi S. |

### 👥 Project Team

- **Nehmal N** – 25MCA043
- **Nidhish Chandran** – 25MCA044

---

## 📌 Project Vision

Venom Watch aims to combine artificial intelligence, geographical analysis, and community-driven reporting to create a smarter approach to snake incident monitoring and public safety.

**Identify. Report. Monitor. Respond.**

---

*Developed as an MCA academic project at TKM College of Engineering, Kollam, Kerala, India.*
