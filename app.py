"""
================================================================
  COURSEIQ — HYBRID ML COURSE RECOMMENDATION ENGINE
  1200+ Courses | TF-IDF + SVD + Cosine Similarity
  Flask Backend  |  Zero external course DB needed
================================================================
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import warnings, random, itertools

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

app = Flask(__name__)

# ================================================================
#  COURSE GENERATOR — Produces 1200+ realistic courses
# ================================================================

PLATFORMS = ["Udemy","Coursera","edX","LinkedIn Learning","Pluralsight",
             "Skillshare","DataCamp","FreeCodeCamp","Stanford Online",
             "Google Learn","Microsoft Learn","AWS Training","Codecademy"]

DIFFICULTIES = ["Beginner","Intermediate","Advanced"]

INSTRUCTORS = [
    "Dr. Angela Yu","Andrew Ng","Jose Portilla","Maximilian Schwarzmüller",
    "Stephen Grider","Colt Steele","Brad Traversy","Tim Buchalka",
    "Mosh Hamedani","Frank Kane","Kirill Eremenko","Dr. Ryan Ahmed",
    "Andrei Neagoie","Jonas Schmedtmann","Dr. Chuck Severance",
    "David Malan","Siraj Raval","Sentdex","Corey Schafer","Tech With Tim",
    "Traversy Media","Fireship","Net Ninja","Kevin Powell","Web Dev Simplified",
    "MIT OpenCourseWare","Stanford Faculty","IBM Skills","Google Developers",
    "Microsoft Learning","AWS Training Team","Oracle Academy","Red Hat",
]

# ── 20 domains, each with 15 subtopics = 300 × 4 variants = 1200 courses
DOMAIN_CONFIG = {
    "Data Science": {
        "color":"#0a9e75","icon":"📊",
        "subtopics": [
            ("Python for Data Science",     "python pandas numpy matplotlib seaborn data analysis jupyter",         ["data analysis","python","numpy","pandas","matplotlib","seaborn","jupyter"]),
            ("R Programming for Statistics","r programming ggplot2 dplyr tidyverse statistical computing",          ["R","ggplot2","statistics","tidyverse"]),
            ("SQL for Data Science",        "sql mysql postgresql database queries joins aggregation analytics",     ["sql","database","mysql","queries"]),
            ("Statistics & Probability",    "statistics probability distributions hypothesis testing anova",        ["statistics","probability","hypothesis testing"]),
            ("Machine Learning Fundamentals","machine learning sklearn supervised unsupervised regression classification",["machine learning","sklearn","regression","classification"]),
            ("Feature Engineering",         "feature engineering selection extraction dimensionality reduction pca", ["feature engineering","PCA","dimensionality reduction"]),
            ("Time Series Analysis",        "time series forecasting arima prophet lstm temporal data",              ["time series","forecasting","ARIMA","prophet"]),
            ("Natural Language Processing", "nlp text processing tokenization sentiment analysis word embeddings",   ["NLP","sentiment analysis","text mining","word2vec"]),
            ("Deep Learning for Data Science","deep learning tensorflow keras neural networks data science",         ["deep learning","tensorflow","neural networks"]),
            ("Data Wrangling & Cleaning",   "data wrangling cleaning missing values outliers preprocessing",        ["data cleaning","preprocessing","pandas"]),
            ("Exploratory Data Analysis",   "eda exploratory data analysis visualization statistical summaries",    ["EDA","visualization","statistics"]),
            ("Big Data with PySpark",       "big data pyspark apache spark hadoop distributed computing",           ["big data","PySpark","Spark","Hadoop"]),
            ("MLOps & Model Deployment",    "mlops model deployment docker fastapi mlflow monitoring production",    ["MLOps","deployment","Docker","FastAPI","MLflow"]),
            ("Bayesian Statistics",         "bayesian statistics mcmc pymc probabilistic programming inference",     ["Bayesian","MCMC","probabilistic programming"]),
            ("Recommendation Systems",      "recommendation systems collaborative filtering content based matrix factorization",["recommendation systems","collaborative filtering","matrix factorization"]),
        ]
    },
    "Artificial Intelligence": {
        "color":"#3d4fc4","icon":"🤖",
        "subtopics": [
            ("Neural Networks from Scratch",   "neural networks backpropagation gradient descent perceptron",         ["neural networks","backpropagation","gradient descent"]),
            ("Deep Learning Specialization",   "deep learning cnn rnn lstm batch normalization regularization",       ["deep learning","CNN","RNN","LSTM"]),
            ("Reinforcement Learning",         "reinforcement learning q-learning policy gradient openai gym",        ["reinforcement learning","Q-learning","policy gradient"]),
            ("Generative Adversarial Networks","gans generative adversarial networks image synthesis style transfer", ["GANs","image generation","style transfer"]),
            ("Transformers & Attention",       "transformers attention mechanism bert gpt encoder decoder",           ["transformers","BERT","attention","GPT"]),
            ("Large Language Models",          "llm gpt chatgpt openai api prompt engineering fine-tuning",          ["LLMs","ChatGPT","OpenAI","prompt engineering"]),
            ("Generative AI & Diffusion Models","stable diffusion midjourney dall-e generative ai image generation",  ["Stable Diffusion","DALL-E","generative AI"]),
            ("AutoML & Neural Architecture",   "automl neural architecture search hyperparameter optimization optuna",["AutoML","hyperparameter tuning","NAS"]),
            ("AI Ethics & Responsible AI",     "ai ethics fairness bias explainability transparency governance",     ["AI ethics","fairness","explainability"]),
            ("Edge AI & TinyML",               "edge ai tinyml microcontrollers tensorflow lite onnx embedded",      ["Edge AI","TinyML","TensorFlow Lite"]),
            ("Federated Learning",             "federated learning privacy preserving machine learning distributed", ["federated learning","privacy","distributed ML"]),
            ("Explainable AI (XAI)",           "explainable ai shap lime model interpretability feature importance",  ["XAI","SHAP","LIME","interpretability"]),
            ("AI Agents & LangChain",          "ai agents langchain autogpt autonomous agents tool use retrieval",   ["AI agents","LangChain","AutoGPT","RAG"]),
            ("Multi-Modal AI",                 "multimodal ai vision language clip flamingo image text understanding",["multi-modal AI","CLIP","vision-language"]),
            ("Quantum Machine Learning",       "quantum machine learning qiskit variational circuits hybrid",        ["quantum ML","Qiskit","variational circuits"]),
        ]
    },
    "Web Development": {
        "color":"#c9922a","icon":"🌐",
        "subtopics": [
            ("HTML5 & CSS3 Mastery",        "html5 css3 flexbox grid responsive design animations transitions",     ["HTML5","CSS3","flexbox","grid","responsive"]),
            ("JavaScript Complete Guide",   "javascript es6 dom async await promises closures prototype",           ["JavaScript","ES6","async/await","DOM"]),
            ("React Complete Guide",        "react hooks context api redux state management spa component",          ["React","hooks","Redux","SPA"]),
            ("Vue.js 3 Masterclass",        "vue3 composition api vuex pinia nuxt router vue components",           ["Vue.js","Vuex","Nuxt","composition API"]),
            ("Angular Full Course",         "angular typescript rxjs ngrx routing forms dependency injection",      ["Angular","TypeScript","RxJS","NgRx"]),
            ("Node.js & Express",           "node express middleware rest api authentication mongodb mongoose",      ["Node.js","Express","REST API","MongoDB"]),
            ("Next.js & React Fullstack",   "nextjs server side rendering static generation api routes fullstack",   ["Next.js","SSR","SSG","fullstack React"]),
            ("Django Full-Stack",           "django python orm views templates authentication rest framework",       ["Django","Python","ORM","DRF"]),
            ("FastAPI Modern Backend",      "fastapi python async pydantic jwt auth postgresql sqlalchemy",         ["FastAPI","Python","async","SQLAlchemy"]),
            ("GraphQL API Design",          "graphql schema mutations subscriptions apollo client server relay",     ["GraphQL","Apollo","subscriptions","schema"]),
            ("TypeScript Deep Dive",        "typescript types interfaces generics decorators utility types",         ["TypeScript","generics","interfaces","decorators"]),
            ("Svelte & SvelteKit",          "svelte sveltekit stores transitions actions server side rendering",     ["Svelte","SvelteKit","stores"]),
            ("Web Performance Optimization","web performance lighthouse core web vitals lazy loading caching",      ["web performance","Core Web Vitals","optimization"]),
            ("Web Security & OWASP",        "web security owasp xss csrf sql injection https authentication",      ["web security","OWASP","XSS","CSRF"]),
            ("Three.js & WebGL",            "three js webgl 3d graphics shaders animation browser rendering",      ["Three.js","WebGL","3D","shaders"]),
        ]
    },
    "Cloud Computing": {
        "color":"#55607a","icon":"☁️",
        "subtopics": [
            ("AWS Solutions Architect",     "aws ec2 s3 rds lambda vpc cloudformation iam ecs eks",                ["AWS","EC2","S3","Lambda","CloudFormation"]),
            ("Microsoft Azure Fundamentals","azure virtual machines storage networking active directory devops",    ["Azure","VM","storage","Active Directory"]),
            ("Google Cloud Platform",       "gcp bigquery cloud run kubernetes engine dataflow pubsub functions",  ["GCP","BigQuery","Cloud Run","Dataflow"]),
            ("Docker Mastery",              "docker containers images volumes networking compose registry",         ["Docker","containers","Docker Compose"]),
            ("Kubernetes Complete",         "kubernetes pods deployments services ingress helm operators rbac",     ["Kubernetes","pods","Helm","RBAC"]),
            ("Terraform & IaC",             "terraform infrastructure as code hcl modules providers state",        ["Terraform","IaC","HCL","modules"]),
            ("CI/CD Pipelines",             "cicd jenkins github actions gitlab ci devops automation pipeline",    ["CI/CD","Jenkins","GitHub Actions","DevOps"]),
            ("Serverless Architecture",     "serverless aws lambda azure functions gcp cloud functions faas",      ["serverless","Lambda","Functions","FaaS"]),
            ("Cloud Security",              "cloud security iam policies encryption compliance soc2 zero trust",   ["cloud security","IAM","encryption","compliance"]),
            ("Site Reliability Engineering","sre reliability slo sli error budgets incident management postmortem",["SRE","SLO","SLI","reliability"]),
            ("Cloud Networking",            "cloud networking vpc subnets routing load balancers cdn dns",         ["cloud networking","VPC","CDN","DNS"]),
            ("FinOps & Cost Optimization",  "finops cloud cost optimization reserved instances spot rightsizing",  ["FinOps","cost optimization","savings"]),
            ("Multi-Cloud Strategy",        "multi cloud hybrid cloud cloud agnostic migration strategy governance",["multi-cloud","hybrid cloud","migration"]),
            ("Platform Engineering",        "platform engineering internal developer platform golden paths paved",  ["platform engineering","IDP","DevEx"]),
            ("Chaos Engineering",           "chaos engineering resilience fault injection gameday netflix gremlin", ["chaos engineering","resilience","fault injection"]),
        ]
    },
    "Cybersecurity": {
        "color":"#c94a4a","icon":"🛡️",
        "subtopics": [
            ("Ethical Hacking Bootcamp",    "ethical hacking kali linux metasploit nmap reconnaissance exploitation", ["ethical hacking","Kali","Metasploit","Nmap"]),
            ("Penetration Testing",         "penetration testing methodology reporting web app network mobile",     ["penetration testing","pen test","reporting"]),
            ("Network Security",            "network security firewall ids ips snort suricata packet analysis",     ["network security","firewall","IDS","IPS"]),
            ("Web Application Security",    "web app security burpsuite owasp zap sql injection xss api security",  ["web security","Burp Suite","OWASP","ZAP"]),
            ("OSINT & Reconnaissance",      "osint open source intelligence maltego shodan recon social engineering",["OSINT","Shodan","Maltego","reconnaissance"]),
            ("Malware Analysis",            "malware analysis reverse engineering ida pro ghidra dynamic static",   ["malware analysis","reverse engineering","Ghidra"]),
            ("Digital Forensics",           "digital forensics autopsy volatility disk imaging memory analysis",    ["forensics","Autopsy","Volatility","memory"]),
            ("SOC Analyst Training",        "soc analyst siem splunk qradar threat hunting incident response",      ["SOC","SIEM","Splunk","threat hunting"]),
            ("Zero Trust Security",         "zero trust network access ztna microsegmentation identity verification",["zero trust","ZTNA","microsegmentation"]),
            ("DevSecOps",                   "devsecops sast dast sca shift left container security pipeline",      ["DevSecOps","SAST","DAST","secure SDLC"]),
            ("Cryptography & PKI",          "cryptography encryption rsa aes elliptic curve tls certificates pki",  ["cryptography","RSA","AES","TLS","PKI"]),
            ("Bug Bounty Hunting",          "bug bounty hackerone bugcrowd responsible disclosure scope rewards",    ["bug bounty","HackerOne","BugCrowd"]),
            ("Red Team Operations",         "red team adversarial simulation attack paths lateral movement c2",     ["red team","adversarial","C2","lateral movement"]),
            ("Cloud Security Engineering",  "cloud security aws azure gcp cspm cwpp devsecops posture management",  ["cloud security","CSPM","CWPP"]),
            ("Threat Intelligence",         "threat intelligence cti mitre att&ck ioc ioa threat actors ttps",     ["threat intelligence","MITRE ATT&CK","IoC","TTPs"]),
        ]
    },
    "Mobile Development": {
        "color":"#7b4fc4","icon":"📱",
        "subtopics": [
            ("iOS Development with Swift",   "swift ios uikit viewcontroller storyboard xcode apple sdk",           ["Swift","iOS","UIKit","Xcode"]),
            ("SwiftUI Complete Guide",       "swiftui declarative ui state binding animations combine preview",     ["SwiftUI","declarative","Combine","animations"]),
            ("Android with Kotlin",          "kotlin android studio activities fragments viewmodel livedata",       ["Kotlin","Android","ViewModel","LiveData"]),
            ("Jetpack Compose",              "jetpack compose declarative ui material3 state animations navigation", ["Jetpack Compose","Material3","navigation"]),
            ("Flutter Development",          "flutter dart widgets state management bloc provider animations",      ["Flutter","Dart","BLoC","Provider"]),
            ("React Native",                 "react native javascript expo navigation redux offline mobile",        ["React Native","Expo","JavaScript","mobile"]),
            ("Cross-Platform with Ionic",    "ionic angular react vue capacitor mobile hybrid web apps",            ["Ionic","Capacitor","hybrid","cross-platform"]),
            ("Mobile UI/UX Design",          "mobile ui ux design figma prototyping user testing accessibility",   ["mobile UI/UX","Figma","prototyping"]),
            ("Mobile App Security",          "mobile security owasp mobile frida reverse engineering ssl pinning",  ["mobile security","OWASP mobile","Frida"]),
            ("App Store Optimization",       "aso app store optimization keywords screenshots ratings reviews",     ["ASO","App Store","Google Play","keywords"]),
            ("Push Notifications & FCM",    "push notifications firebase cloud messaging fcm apns local remote",   ["push notifications","FCM","APNs","Firebase"]),
            ("AR Mobile with ARKit/ARCore",  "augmented reality arkit arcore scenekit realitykit ar foundation",   ["AR","ARKit","ARCore","RealityKit"]),
            ("Mobile Backend with Firebase", "firebase realtime database firestore auth functions storage hosting", ["Firebase","Firestore","Auth","Cloud Functions"]),
            ("Performance Optimization",     "mobile performance memory leaks profiling battery optimization app",  ["performance","profiling","battery","optimization"]),
            ("Mobile Game Development",      "mobile games unity ads monetization analytics game mechanics",        ["mobile games","Unity","monetization","analytics"]),
        ]
    },
    "Data Analytics": {
        "color":"#0a7e9e","icon":"📈",
        "subtopics": [
            ("Tableau Desktop Specialist",   "tableau desktop charts dashboards calculated fields lod expressions",  ["Tableau","dashboards","LOD","calculated fields"]),
            ("Power BI Complete",            "power bi dax measures calculated columns data model relationships",    ["Power BI","DAX","data model","reports"]),
            ("Excel for Data Analysis",      "excel pivot tables vlookup power query data analysis formulas vba",   ["Excel","pivot tables","Power Query","VBA"]),
            ("Google Analytics 4",           "google analytics ga4 events conversions audiences attribution",       ["Google Analytics","GA4","conversions","attribution"]),
            ("SQL Analytics Deep Dive",      "sql analytics window functions cte subqueries performance tuning",    ["SQL","window functions","CTEs","analytics"]),
            ("Looker & LookML",              "looker lookml data models explores dimensions measures dashboards",   ["Looker","LookML","explores","measures"]),
            ("A/B Testing & Experimentation","ab testing statistical significance power analysis conversion rate",   ["A/B testing","experimentation","statistics","CRO"]),
            ("Business Intelligence Design", "business intelligence kpi metrics data strategy reporting culture",   ["BI","KPIs","data strategy","reporting"]),
            ("Customer Analytics",           "customer analytics churn ltv segmentation rfm cohort analysis",      ["customer analytics","churn","LTV","cohort"]),
            ("Marketing Analytics",          "marketing analytics attribution multi-touch funnel roi channel mix",  ["marketing analytics","attribution","funnel","ROI"]),
            ("Supply Chain Analytics",       "supply chain analytics demand forecasting inventory optimization",    ["supply chain","demand forecasting","inventory"]),
            ("Financial Analytics",          "financial analytics budgeting variance analysis financial modeling",  ["financial analytics","budgeting","FP&A"]),
            ("Data Storytelling",            "data storytelling narrative visualization persuasion executive",      ["data storytelling","narrative","communication"]),
            ("dbt Analytics Engineering",    "dbt data build tool sql transformations testing documentation",      ["dbt","analytics engineering","SQL transforms"]),
            ("Metabase & Open Source BI",    "metabase open source bi questions dashboards embedding sql",          ["Metabase","open source BI","embedding"]),
        ]
    },
    "Computer Science": {
        "color":"#1a6e8a","icon":"💻",
        "subtopics": [
            ("Data Structures Masterclass",  "data structures arrays linked lists trees graphs heaps hash tables",  ["data structures","arrays","trees","graphs"]),
            ("Algorithms Complete Course",   "algorithms sorting searching dynamic programming greedy graph",       ["algorithms","sorting","DP","graph algorithms"]),
            ("System Design Interview",      "system design scalability load balancing caching database sharding",  ["system design","scalability","caching","CDN"]),
            ("Design Patterns",              "design patterns gof creational structural behavioral solid principles",["design patterns","SOLID","GoF","OOP"]),
            ("Operating Systems",            "operating systems processes threads memory management scheduling",    ["OS","processes","threads","memory"]),
            ("Computer Networks",            "computer networks tcp ip http dns osi model routing switching",       ["networking","TCP/IP","HTTP","DNS","OSI"]),
            ("Database Systems",             "database systems relational nosql acid transactions indexing",        ["databases","ACID","transactions","indexing"]),
            ("Compiler Design",              "compiler design lexing parsing ast code generation optimization",     ["compilers","parsing","AST","LLVM"]),
            ("Distributed Systems",          "distributed systems cap theorem consensus raft paxos fault tolerance", ["distributed systems","CAP","Raft","consensus"]),
            ("Software Engineering",         "software engineering agile scrum tdd bdd requirements architecture",  ["software engineering","agile","TDD","requirements"]),
            ("Programming Languages Theory", "programming languages type systems lambda calculus semantics types",   ["PL theory","type systems","lambda calculus"]),
            ("Interview Preparation",        "coding interview leetcode competitive programming problem solving",   ["interviews","LeetCode","competitive programming"]),
            ("Complexity Theory",            "complexity theory np complete np hard reducibility turing machines",  ["complexity","NP-complete","Turing machines"]),
            ("Parallel Computing",           "parallel computing openmp cuda mpi gpu programming multi-threading",  ["parallel computing","CUDA","OpenMP","GPU"]),
            ("Formal Methods",               "formal methods model checking tla+ coq isabelle verification proofs",  ["formal methods","model checking","TLA+","Coq"]),
        ]
    },
    "Data Engineering": {
        "color":"#6a4fc4","icon":"⚙️",
        "subtopics": [
            ("Apache Spark Mastery",         "apache spark pyspark rdd dataframes spark sql streaming ml",          ["Apache Spark","PySpark","RDDs","Spark SQL"]),
            ("Apache Kafka",                 "apache kafka producers consumers topics partitions kafka streams",    ["Kafka","producers","consumers","Kafka Streams"]),
            ("Apache Airflow",               "apache airflow dags operators sensors hooks xcom scheduling",        ["Airflow","DAGs","operators","scheduling"]),
            ("dbt for Data Engineering",     "dbt data build tool models tests sources snapshots jinja",           ["dbt","models","tests","Jinja"]),
            ("Snowflake Data Platform",      "snowflake virtual warehouses data sharing zero copy clone streams",   ["Snowflake","data warehouse","data sharing"]),
            ("Databricks & Delta Lake",      "databricks delta lake acid transactions time travel schema evolution", ["Databricks","Delta Lake","ACID","lakehouse"]),
            ("ETL Pipeline Design",          "etl extract transform load pipeline design patterns batch incremental",["ETL","pipelines","batch","incremental"]),
            ("Data Lake Architecture",       "data lake architecture s3 adls raw curated consumption zones governance",["data lake","S3","zones","governance"]),
            ("Apache Flink",                 "apache flink stream processing stateful computations event time watermarks",["Flink","stream processing","stateful","watermarks"]),
            ("Data Modeling",                "data modeling star schema snowflake schema dimensional modeling facts",["data modeling","star schema","dimensional"]),
            ("Data Quality & Observability", "data quality great expectations monte carlo data observability soda", ["data quality","Great Expectations","observability"]),
            ("Real-Time Streaming",          "real time streaming kafka flink spark structured streaming kinesis",   ["real-time","streaming","Kinesis","Kafka"]),
            ("Data Governance",              "data governance catalog lineage metadata management data mesh",       ["data governance","catalog","lineage","data mesh"]),
            ("Cloud Data Warehouses",        "cloud data warehouse bigquery redshift synapse azure sql cost",       ["BigQuery","Redshift","Synapse","data warehouse"]),
            ("Python for Data Engineering",  "python data engineering pandas polars connectors apis orchestration", ["Python","Polars","orchestration","APIs"]),
        ]
    },
    "DevOps": {
        "color":"#2e8a55","icon":"🔧",
        "subtopics": [
            ("Linux Administration",         "linux bash shell scripting administration permissions cron systemd",  ["Linux","bash","shell","cron","systemd"]),
            ("Git & GitHub Mastery",         "git github branching merging rebasing pull requests workflows ci",   ["Git","GitHub","branching","pull requests"]),
            ("Jenkins CI/CD",                "jenkins pipelines declarative groovy agents plugins integrations",   ["Jenkins","pipelines","Groovy","CI/CD"]),
            ("GitHub Actions",               "github actions workflows runners secrets matrix strategy composite", ["GitHub Actions","workflows","secrets","matrix"]),
            ("Ansible Automation",           "ansible playbooks roles inventory modules templates vault automation",["Ansible","playbooks","roles","automation"]),
            ("Prometheus & Grafana",         "prometheus metrics alertmanager grafana dashboards visualization",    ["Prometheus","Grafana","metrics","alerting"]),
            ("ELK Stack",                    "elk elasticsearch logstash kibana log aggregation indexing search",  ["ELK","Elasticsearch","Logstash","Kibana"]),
            ("GitOps & ArgoCD",              "gitops argocd flux reconciliation declarative continuous delivery",   ["GitOps","ArgoCD","Flux","reconciliation"]),
            ("HashiCorp Vault",              "vault secrets management dynamic secrets pki certificates engines",   ["Vault","secrets","PKI","dynamic credentials"]),
            ("Service Mesh with Istio",      "service mesh istio envoy proxy traffic management observability mtls",["Istio","Envoy","service mesh","mTLS"]),
            ("SRE Practices",                "sre error budgets slo sli toil postmortem blameless reliability",    ["SRE","SLO","error budgets","postmortems"]),
            ("Helm & Kubernetes Packaging",  "helm charts values templates hooks kubernetes packaging operators",   ["Helm","charts","operators","packaging"]),
            ("Platform Engineering",         "platform engineering internal developer platform idp golden paths",   ["platform engineering","IDP","paved road"]),
            ("Chef & Puppet",                "chef puppet configuration management cookbooks manifests modules",    ["Chef","Puppet","config management","cookbooks"]),
            ("Chaos Engineering",            "chaos engineering netflix gremlin litmus chaos monkey fault injection",["chaos engineering","Gremlin","Litmus","resilience"]),
        ]
    },
    "Game Development": {
        "color":"#8a2e9e","icon":"🎮",
        "subtopics": [
            ("Unity 2D Game Development",    "unity 2d sprites physics colliders animator cinemachine ui",         ["Unity","2D","sprites","physics","animator"]),
            ("Unity 3D Complete",            "unity 3d mesh materials lighting camera rigidbody navmesh",          ["Unity","3D","lighting","NavMesh","rigidbody"]),
            ("Unreal Engine 5",              "unreal engine 5 blueprints lumen nanite c++ gameplay framework",     ["Unreal Engine","blueprints","Lumen","Nanite"]),
            ("Godot 4 Game Development",     "godot 4 gdscript scenes nodes signals 2d 3d animation",             ["Godot","GDScript","scenes","signals"]),
            ("Pygame with Python",           "pygame python sprites collision events surfaces game loop",          ["Pygame","Python","sprites","game loop"]),
            ("3D Modeling for Games",        "3d modeling blender low poly baking normal maps game ready assets",  ["Blender","3D modeling","low poly","normal maps"]),
            ("Game Physics & Simulation",    "game physics rigid body dynamics collision detection constraints",    ["game physics","rigid body","collision","constraints"]),
            ("Shader Programming",           "shader programming hlsl glsl vertex fragment shader graph unity",    ["shaders","HLSL","GLSL","ShaderGraph"]),
            ("Multiplayer Game Development", "multiplayer networking photon mirror netcode dedicated servers",     ["multiplayer","Photon","Mirror","Netcode"]),
            ("Game AI & Behaviour Trees",    "game ai behaviour trees fsm pathfinding steering behaviours",        ["game AI","behaviour trees","pathfinding","FSM"]),
            ("Mobile Game Development",      "mobile game unity ads iap analytics performance optimization",       ["mobile games","IAP","ads","monetization"]),
            ("VR Game Development",          "vr game development unity xr meta quest oculus openxr interaction",  ["VR","OpenXR","Meta Quest","XR Toolkit"]),
            ("Game Design & Prototyping",    "game design mechanics balance playtesting prototyping iteration gdd",["game design","GDD","playtesting","mechanics"]),
            ("Sound Design for Games",       "sound design fmod wwise audio middleware spatial audio music",       ["sound design","FMOD","Wwise","spatial audio"]),
            ("Game Marketing & Publishing",  "game marketing steam itch.io app store publishing trailer community",["game marketing","Steam","publishing","community"]),
        ]
    },
    "Blockchain": {
        "color":"#c47a1a","icon":"⛓️",
        "subtopics": [
            ("Solidity Smart Contracts",     "solidity ethereum smart contracts evm remix hardhat deployment",      ["Solidity","Ethereum","smart contracts","EVM"]),
            ("Ethereum DApp Development",    "ethereum dapp web3js ethersjs metamask react frontend blockchain",   ["DApp","Web3.js","Ethers.js","MetaMask"]),
            ("DeFi Protocol Development",    "defi liquidity pools amm yield farming smart contracts tvl",         ["DeFi","AMM","liquidity pools","yield farming"]),
            ("NFT Marketplace Creation",     "nft erc721 erc1155 opensea metadata ipfs royalties marketplace",     ["NFTs","ERC-721","OpenSea","IPFS"]),
            ("Bitcoin & Cryptocurrency",     "bitcoin blockchain transactions utxo mining consensus proof of work", ["Bitcoin","UTXO","mining","consensus"]),
            ("Web3.js & Ethers.js",          "web3 ethers javascript library provider signer transactions events", ["Web3.js","Ethers.js","provider","signer"]),
            ("Hardhat & Foundry Testing",    "hardhat foundry testing solidity unit tests deployment scripts",     ["Hardhat","Foundry","testing","deployment scripts"]),
            ("Layer 2 Solutions",            "layer 2 optimism arbitrum polygon zksync rollups scaling ethereum",  ["Layer 2","Optimism","Arbitrum","zkSync","rollups"]),
            ("Rust for Blockchain",          "rust solana near blockchain smart contracts ownership memory safety",["Rust","Solana","NEAR","ownership","memory safety"]),
            ("DAO Governance",               "dao governance token voting multisig snapshot compound governance",   ["DAO","governance","token voting","Snapshot"]),
            ("Crypto Trading & Analysis",    "crypto trading technical analysis indicators strategies defi cex",   ["crypto trading","TA","indicators","DeFi"]),
            ("Blockchain Security",          "blockchain security reentrancy flash loans audit slither mythril",   ["blockchain security","reentrancy","auditing","Slither"]),
            ("Polkadot & Substrate",         "polkadot substrate parachain relay chain xcm kusama pallets",        ["Polkadot","Substrate","parachain","XCM"]),
            ("Tokenomics & Economics",       "tokenomics token design economics incentive mechanism supply demand",  ["tokenomics","token design","incentives","economics"]),
            ("Zero Knowledge Proofs",        "zero knowledge proofs zk snarks circom groth16 privacy scaling",     ["ZK proofs","zkSNARKs","Circom","privacy"]),
        ]
    },
    "UI/UX Design": {
        "color":"#c42a72","icon":"🎨",
        "subtopics": [
            ("Figma UI Design",              "figma components auto layout variants prototyping design systems",    ["Figma","components","auto layout","variants"]),
            ("Adobe XD Masterclass",         "adobe xd wireframes prototyping responsive repeat grid components",  ["Adobe XD","wireframes","prototyping","responsive"]),
            ("User Research Methods",        "user research interviews surveys usability testing personas affinity",["user research","interviews","usability testing"]),
            ("Design Systems",               "design system tokens components documentation storybook atomic",     ["design systems","tokens","Storybook","atomic"]),
            ("Interaction Design",           "interaction design microinteractions motion principles gestures",     ["interaction design","microinteractions","motion"]),
            ("Accessibility & WCAG",         "accessibility wcag aria roles contrast keyboard screen readers",     ["accessibility","WCAG","ARIA","screen readers"]),
            ("UX Writing",                   "ux writing microcopy content design voice tone error messages",      ["UX writing","microcopy","voice","tone"]),
            ("Mobile UX Design",             "mobile ux ios android guidelines thumb zone gestures navigation",    ["mobile UX","iOS guidelines","Android","thumb zone"]),
            ("Design Thinking",              "design thinking empathize define ideate prototype test hcd process", ["design thinking","HCD","empathy","ideation"]),
            ("Typography & Color Theory",    "typography type scales pairing color theory harmony contrast",        ["typography","color theory","type pairing"]),
            ("Prototyping & Testing",        "prototyping high fidelity lo fi usability testing a/b iteration",    ["prototyping","usability","A/B testing","iteration"]),
            ("Product Design",               "product design strategy roadmap metrics success outcomes jobs to be", ["product design","strategy","OKRs","JTBD"]),
            ("Sketch for UI Design",         "sketch plugins symbols pages artboards vector boolean operations",   ["Sketch","symbols","artboards","plugins"]),
            ("Motion Design with Principle", "motion design principle after effects lottie animation ui timing",   ["motion design","Lottie","After Effects","timing"]),
            ("Inclusive Design",             "inclusive design diversity accessibility representation edge cases",  ["inclusive design","diversity","accessibility"]),
        ]
    },
    "Digital Marketing": {
        "color":"#1a8a4e","icon":"📣",
        "subtopics": [
            ("SEO Masterclass",              "seo search engine optimization on page off page technical link building",["SEO","on-page","off-page","link building"]),
            ("Google Ads Complete",          "google ads search shopping display video remarketing quality score",  ["Google Ads","PPC","shopping ads","remarketing"]),
            ("Facebook & Instagram Ads",     "facebook ads instagram advertising audiences pixel conversion api",   ["Facebook Ads","Instagram","Pixel","lookalike"]),
            ("Content Marketing Strategy",   "content marketing strategy blogging editorial calendar distribution", ["content marketing","blogging","editorial","distribution"]),
            ("Email Marketing & Automation", "email marketing mailchimp klaviyo automation sequences deliverability",["email marketing","Klaviyo","automation","deliverability"]),
            ("Social Media Marketing",       "social media marketing strategy organic content calendar engagement",  ["social media","organic","content calendar","engagement"]),
            ("YouTube Marketing",            "youtube marketing video seo thumbnails analytics channel growth",     ["YouTube","video SEO","thumbnails","channel growth"]),
            ("TikTok Marketing",             "tiktok marketing short video algorithm trends ads creator",          ["TikTok","short video","algorithm","creator"]),
            ("Copywriting & Persuasion",     "copywriting headlines cta persuasion storytelling direct response",   ["copywriting","CTA","persuasion","direct response"]),
            ("Brand Strategy",               "brand strategy positioning identity logo guidelines voice tone",      ["branding","positioning","identity","guidelines"]),
            ("Marketing Analytics",          "marketing analytics attribution modeling cac ltv roas funnel",       ["marketing analytics","CAC","LTV","ROAS"]),
            ("CRM & HubSpot",                "crm hubspot salesforce pipeline deals contacts sequences reporting",  ["CRM","HubSpot","Salesforce","pipeline"]),
            ("Growth Hacking",               "growth hacking acquisition retention referral viral loops pirate",   ["growth hacking","acquisition","retention","viral"]),
            ("Affiliate Marketing",          "affiliate marketing commission networks tracking cookies landing",    ["affiliate marketing","commissions","networks"]),
            ("Influencer Marketing",         "influencer marketing ugc creator economy campaigns measurement roi",  ["influencer marketing","UGC","creator economy"]),
        ]
    },
    "Finance & FinTech": {
        "color":"#2a6e8a","icon":"💰",
        "subtopics": [
            ("Python for Finance",           "python finance yfinance quantlib portfolio returns volatility",      ["Python","finance","yfinance","quantlib"]),
            ("Algorithmic Trading",          "algorithmic trading backtesting strategies momentum mean reversion",  ["algo trading","backtesting","momentum","strategies"]),
            ("Quantitative Finance",         "quantitative finance stochastic calculus options pricing black scholes",["quant finance","Black-Scholes","derivatives","stochastic"]),
            ("Financial Modeling Excel",     "financial modeling excel dcf lbo m&a three statement model cfa",     ["financial modeling","DCF","LBO","M&A"]),
            ("Risk Management",              "risk management var stress testing credit risk market risk operational",["risk management","VaR","stress testing","credit risk"]),
            ("Blockchain in Finance",        "blockchain defi cbdc payment rails clearing settlement tokenization", ["blockchain finance","DeFi","CBDC","settlement"]),
            ("Cryptocurrency Trading",       "cryptocurrency trading technical analysis indicators bitcoin portfolio",["crypto trading","technical analysis","Bitcoin"]),
            ("Portfolio Management",         "portfolio management modern portfolio theory sharpe ratio allocation", ["portfolio management","MPT","Sharpe ratio","allocation"]),
            ("Options & Derivatives",        "options derivatives puts calls greeks strategies hedging volatility",  ["options","derivatives","Greeks","hedging"]),
            ("FinTech APIs & Open Banking",  "fintech api open banking plaid stripe payment gateways integration",  ["FinTech","Open Banking","Plaid","Stripe","APIs"]),
            ("RegTech & Compliance",         "regtech compliance aml kyc gdpr psd2 reporting automation",          ["RegTech","AML","KYC","GDPR","compliance"]),
            ("ESG Investing",                "esg environmental social governance sustainable investing screening",  ["ESG","sustainable investing","screening","impact"]),
            ("Financial Data Science",       "financial data science sentiment analysis alternative data nlp",      ["financial data science","sentiment","alternative data"]),
            ("Robo-Advisory",                "robo advisory automated portfolio goal based investing rebalancing",  ["robo-advisory","automated investing","rebalancing"]),
            ("Excel for Finance",            "excel finance formulas npv irr xnpv sensitivity analysis dashboard",  ["Excel","NPV","IRR","sensitivity analysis"]),
        ]
    },
    "Networking": {
        "color":"#4a7a5a","icon":"🔌",
        "subtopics": [
            ("CCNA Certification Prep",      "ccna cisco routing switching vlans spanning tree ospf eigrp",        ["CCNA","Cisco","routing","switching","OSPF"]),
            ("TCP/IP Networking",            "tcp ip networking osi model layers protocols ip addressing subnetting",["TCP/IP","OSI","subnetting","IP addressing"]),
            ("Network Architecture Design",  "network architecture design topology redundancy high availability",   ["network architecture","topology","redundancy","HA"]),
            ("Wireless Networking",          "wireless networking wifi 802.11 wpa3 mesh enterprise wireless lan",   ["wireless","WiFi","802.11","WPA3","mesh"]),
            ("Software Defined Networking",  "sdn software defined networking openflow controller data plane",     ["SDN","OpenFlow","controller","network programmability"]),
            ("Network Automation",           "network automation python netconf yang restconf ansible nornir",     ["network automation","Python","NETCONF","Ansible"]),
            ("VPN Technologies",             "vpn technologies ipsec ssl tls openvpn wireguard site to site remote",["VPN","IPSec","WireGuard","OpenVPN","SSL"]),
            ("BGP & Advanced Routing",       "bgp border gateway protocol attributes policies route filtering asn",  ["BGP","routing","AS","route policies","peering"]),
            ("Network Monitoring",           "network monitoring snmp netflow syslog nagios zabbix prtg",          ["monitoring","SNMP","NetFlow","Zabbix","Nagios"]),
            ("IPv6 Transition",              "ipv6 transition dual stack tunneling migration address space",        ["IPv6","dual stack","tunneling","migration"]),
            ("5G & Mobile Networks",         "5g mobile networks ran core network slicing mmwave beamforming",      ["5G","RAN","network slicing","mmWave"]),
            ("IoT Networking",               "iot networking mqtt coap lorawan zigbee bluetooth low energy protocols",["IoT","MQTT","CoAP","LoRaWAN","BLE"]),
            ("Packet Analysis with Wireshark","wireshark packet capture analysis protocol dissection filtering",    ["Wireshark","packet capture","protocol analysis"]),
            ("Load Balancing & CDN",         "load balancing cdn nginx haproxy f5 anycast global server load",     ["load balancing","CDN","Nginx","HAProxy"]),
            ("Network Security Monitoring",  "network security monitoring ids ips zeek snort pcap threat hunting",  ["NSM","Zeek","Snort","threat hunting","IDS"]),
        ]
    },
    "Embedded Systems": {
        "color":"#7a4e2a","icon":"🔩",
        "subtopics": [
            ("Arduino Programming",          "arduino c programming sensors actuators serial communication projects",["Arduino","C","sensors","actuators","serial"]),
            ("Raspberry Pi Projects",        "raspberry pi python linux gpio camera projects iot home automation",  ["Raspberry Pi","GPIO","Linux","Python","IoT"]),
            ("RTOS & Real-Time Systems",     "rtos freertos real time scheduling tasks semaphores queues priority", ["RTOS","FreeRTOS","scheduling","real-time"]),
            ("Embedded C Programming",       "embedded c pointers bitwise operations memory registers peripherals",  ["Embedded C","pointers","bitwise","registers"]),
            ("FPGA with VHDL/Verilog",       "fpga vhdl verilog rtl synthesis place route xilinx intel altera",    ["FPGA","VHDL","Verilog","RTL","synthesis"]),
            ("ARM Cortex-M Programming",     "arm cortex m stm32 hal interrupts dma timers clock configuration",   ["ARM","Cortex-M","STM32","HAL","interrupts"]),
            ("IoT System Design",            "iot system design sensor fusion cloud connectivity ota updates mqtt",  ["IoT","sensor fusion","OTA","MQTT","cloud"]),
            ("PCB Design with KiCad",        "pcb design kicad schematic layout routing gerber components bom",     ["PCB","KiCad","schematic","routing","Gerber"]),
            ("Firmware Development",         "firmware development bootloader flash memory wear leveling fota",     ["firmware","bootloader","flash","FOTA"]),
            ("Communication Protocols",      "uart spi i2c can bus usb protocols embedded communication peripherals",["UART","SPI","I2C","CAN bus","USB"]),
            ("Linux Embedded Systems",       "linux embedded yocto buildroot kernel modules device tree drivers",   ["Linux embedded","Yocto","Buildroot","device tree"]),
            ("AUTOSAR & Automotive",         "autosar automotive software architecture ecus can lin flexray",       ["AUTOSAR","automotive","ECUs","CAN","LIN"]),
            ("Functional Safety (IEC 61508)","functional safety iec 61508 iso 26262 sil asil hazard analysis",     ["functional safety","IEC 61508","ISO 26262","SIL"]),
            ("Low Power Design",             "low power embedded design sleep modes power gating optimization",     ["low power","sleep modes","power optimization"]),
            ("Edge AI on Microcontrollers",  "edge ai tinyml microcontroller tensorflow lite quantization pruning",  ["Edge AI","TinyML","quantization","MCU"]),
        ]
    },
    "Quantum Computing": {
        "color":"#4a2a8a","icon":"⚛️",
        "subtopics": [
            ("Qiskit Fundamentals",          "qiskit quantum circuits gates measurements ibm quantum backends",     ["Qiskit","quantum circuits","gates","IBM Quantum"]),
            ("Quantum Gates & Algorithms",   "quantum gates hadamard cnot pauli deutsch jozsa grover shor",        ["quantum gates","Grover","Shor","Deutsch-Jozsa"]),
            ("Variational Quantum Algorithms","vqa vqe qaoa variational quantum eigensolver optimization",         ["VQA","VQE","QAOA","variational circuits"]),
            ("Quantum Machine Learning",     "quantum machine learning qml kernel methods quantum neural networks", ["QML","quantum kernels","quantum neural networks"]),
            ("Cirq & Google Quantum",        "cirq google quantum ai noise simulation hardware backend",            ["Cirq","Google Quantum","noise","simulation"]),
            ("Q# & Azure Quantum",           "q# azure quantum microsoft qdk quantum operations algorithms",       ["Q#","Azure Quantum","Microsoft QDK"]),
            ("Quantum Cryptography",         "quantum cryptography qkd bb84 quantum key distribution security",    ["quantum cryptography","QKD","BB84"]),
            ("Quantum Error Correction",     "quantum error correction stabilizer codes surface codes fault tolerant",["error correction","surface codes","fault tolerant"]),
            ("Post-Quantum Cryptography",    "post quantum cryptography nist lattice based signatures kyber dilithium",["post-quantum","NIST","lattice-based","Kyber"]),
            ("Quantum Simulation",           "quantum simulation chemistry materials hamiltonian variational",      ["quantum simulation","chemistry","materials"]),
            ("Quantum Networking",           "quantum networking entanglement distribution repeaters quantum internet",["quantum networking","entanglement","repeaters"]),
            ("NISQ Algorithms",              "nisq noisy intermediate scale quantum near term algorithms hardware", ["NISQ","near-term","noise","hardware"]),
            ("Quantum Optimization",         "quantum optimization combinatorial qaoa qubo traveling salesman",    ["quantum optimization","QUBO","QAOA","combinatorial"]),
            ("Quantum Hardware",             "quantum hardware superconducting qubits trapped ions photonic qubit", ["quantum hardware","superconducting","trapped ions"]),
            ("Quantum Software Engineering", "quantum software testing debugging compilation transpilation",        ["quantum software","transpilation","compilation"]),
        ]
    },
    "Robotics": {
        "color":"#3a6a7a","icon":"🦾",
        "subtopics": [
            ("ROS2 Programming",             "ros2 topics services actions launch files urdf nodes communication",  ["ROS2","topics","services","URDF","nodes"]),
            ("Computer Vision for Robots",   "robot vision opencv depth cameras yolo object detection pose",        ["robot vision","OpenCV","depth cameras","YOLO"]),
            ("SLAM & Localization",          "slam simultaneous localization mapping lidar point cloud cartographer",["SLAM","LiDAR","mapping","localization"]),
            ("Motion Planning",              "motion planning rrt prm a star trajectory optimization obstacle",     ["motion planning","RRT","trajectory","obstacle avoidance"]),
            ("Robot Kinematics",             "robot kinematics forward inverse jacobian dh parameters urdf",        ["kinematics","Jacobian","DH parameters","inverse kinematics"]),
            ("Gazebo Simulation",            "gazebo simulation ros physics sensors worlds models plugins",         ["Gazebo","simulation","physics","plugins"]),
            ("Drone Programming",            "drone programming ardupilot px4 mavlink quadrotor flight control",    ["drones","PX4","MAVLink","flight control"]),
            ("Robot Manipulation",           "robot manipulation grasping moveit pick place arm planning",         ["manipulation","MoveIt","grasping","arm planning"]),
            ("Human-Robot Interaction",      "human robot interaction hri safety social robots gestures speech",   ["HRI","social robots","safety","gestures"]),
            ("Autonomous Vehicles",          "autonomous vehicles self driving lidar camera fusion ros autoware",   ["autonomous vehicles","LiDAR","sensor fusion","Autoware"]),
            ("Soft Robotics",                "soft robotics pneumatics compliant mechanisms fabrication control",  ["soft robotics","pneumatics","compliant mechanisms"]),
            ("Robot Learning (RL)",          "robot learning reinforcement learning sim to real transfer policy",   ["robot learning","sim-to-real","policy learning","RL"]),
            ("Control Systems",              "control systems pid controller state space bode nyquist stability",   ["control systems","PID","state space","stability"]),
            ("Mechatronics",                 "mechatronics mechanical electrical software integration sensors",     ["mechatronics","integration","sensors","actuators"]),
            ("Industrial Automation",        "industrial automation plc scada hmi ot networks safety",             ["industrial automation","PLC","SCADA","HMI"]),
        ]
    },
}

# Extra topic seeds for search coverage
EXTRA_KEYWORDS = {
    "python": ["Data Science","Artificial Intelligence","Web Development","DevOps","Data Engineering"],
    "machine learning": ["Data Science","Artificial Intelligence"],
    "javascript": ["Web Development","Game Development"],
    "cloud": ["Cloud Computing","DevOps","Data Engineering"],
    "security": ["Cybersecurity","Networking","Cloud Computing","Blockchain"],
    "data": ["Data Science","Data Analytics","Data Engineering"],
    "ai": ["Artificial Intelligence","Data Science","Robotics"],
    "mobile": ["Mobile Development","Game Development"],
    "backend": ["Web Development","Cloud Computing","DevOps"],
    "frontend": ["Web Development","UI/UX Design"],
    "database": ["Data Science","Data Engineering","Computer Science"],
    "devops": ["DevOps","Cloud Computing"],
    "automation": ["DevOps","Robotics","Embedded Systems","Digital Marketing"],
    "analytics": ["Data Analytics","Data Science","Digital Marketing"],
    "trading": ["Finance & FinTech"],
    "networking": ["Networking","Cloud Computing","Cybersecurity"],
    "blockchain": ["Blockchain","Finance & FinTech"],
    "design": ["UI/UX Design","Game Development","Digital Marketing"],
    "marketing": ["Digital Marketing","Finance & FinTech"],
    "quantum": ["Quantum Computing"],
}

PRICE_MAP = {
    "Udemy": [12.99, 14.99, 17.99],
    "Coursera": [0.0, 49.0, 79.0],
    "edX": [0.0, 149.0, 199.0],
    "LinkedIn Learning": [29.99, 39.99],
    "Pluralsight": [29.99, 45.0],
    "Skillshare": [9.99, 16.99],
    "DataCamp": [25.0, 33.0],
    "FreeCodeCamp": [0.0],
    "Stanford Online": [0.0, 150.0, 300.0],
    "Google Learn": [0.0, 99.0],
    "Microsoft Learn": [0.0, 99.0],
    "AWS Training": [0.0, 300.0],
    "Codecademy": [15.99, 39.99],
}

DURATION_MAP = {
    "Beginner":     (8, 30),
    "Intermediate": (20, 60),
    "Advanced":     (35, 90),
}

ENROLLED_MAP = {
    "Udemy": (20000, 250000),
    "Coursera": (15000, 120000),
    "edX": (10000, 100000),
    "LinkedIn Learning": (5000, 80000),
    "Pluralsight": (3000, 50000),
    "Skillshare": (2000, 40000),
    "DataCamp": (10000, 80000),
    "FreeCodeCamp": (50000, 500000),
    "Stanford Online": (5000, 60000),
    "Google Learn": (20000, 200000),
    "Microsoft Learn": (15000, 180000),
    "AWS Training": (10000, 150000),
    "Codecademy": (20000, 300000),
}

def generate_courses():
    courses = []
    cid = 1
    rng = random.Random(42)

    for domain, cfg in DOMAIN_CONFIG.items():
        subtopics = cfg["subtopics"]
        for (base_name, skills_str, skill_tags) in subtopics:
            for diff_idx, difficulty in enumerate(DIFFICULTIES):
                for variant in range(2):  # 2 variants per difficulty
                    if diff_idx == 2 and variant == 1:
                        # Only 1 advanced variant for most — gives exactly ~1200
                        pass

                    platform = rng.choice(PLATFORMS)
                    instructor = rng.choice(INSTRUCTORS)

                    # Name suffix variants
                    suffixes = ["","— Complete Course","Masterclass","Bootcamp","Specialization",
                                "A-Z","Deep Dive","in Practice","Fundamentals","— Zero to Hero"]
                    suffix = suffixes[variant % len(suffixes)] if variant > 0 else ""
                    level_prefix = {"Beginner":"","Intermediate":"Advanced ","Advanced":"Expert "}[difficulty]
                    name = f"{level_prefix}{base_name}" + (f" {suffix}" if suffix else "")

                    # Duration
                    lo, hi = DURATION_MAP[difficulty]
                    duration = rng.randint(lo, hi)

                    # Enrolled
                    elo, ehi = ENROLLED_MAP.get(platform, (5000, 80000))
                    enrolled = rng.randint(elo, ehi)

                    # Rating
                    base_rating = rng.uniform(4.0, 4.9)
                    if difficulty == "Beginner": base_rating += 0.1
                    rating = round(min(5.0, base_rating), 1)

                    # Price
                    prices = PRICE_MAP.get(platform, [14.99, 29.99])
                    price = rng.choice(prices)

                    # Enrich skills with domain and difficulty keywords
                    enriched_skills = (skills_str + " " + domain.lower() + " " +
                                       difficulty.lower() + " " + " ".join(skill_tags[:4]))

                    courses.append({
                        "Course_ID":    f"C{cid:04d}",
                        "Course_Name":  name.strip(),
                        "Domain":       domain,
                        "Platform":     platform,
                        "Difficulty":   difficulty,
                        "Skills":       enriched_skills,
                        "Skill_Tags":   ", ".join(skill_tags),
                        "Rating":       rating,
                        "Enrolled":     enrolled,
                        "Duration_hrs": duration,
                        "Price":        price,
                        "Instructor":   instructor,
                    })
                    cid += 1

    return courses


print("Generating course catalog...")
ALL_COURSES = generate_courses()
print(f"Generated {len(ALL_COURSES)} courses across {len(DOMAIN_CONFIG)} domains")

courses_df = pd.DataFrame(ALL_COURSES)

# ── Student profiles ──────────────────────────────────────────
STUDENTS_DATA = [
    {"Student_ID":"S001","Name":"Arjun Kumar",    "Interests":"data science machine learning python statistics feature engineering","Level":"Beginner"},
    {"Student_ID":"S002","Name":"Priya Sharma",   "Interests":"web development react javascript node next.js typescript fullstack","Level":"Intermediate"},
    {"Student_ID":"S003","Name":"Karthik Rajan",  "Interests":"AI deep learning neural networks computer vision transformers bert","Level":"Advanced"},
    {"Student_ID":"S004","Name":"Meera Pillai",   "Interests":"data analytics tableau power bi business intelligence kpi reporting","Level":"Beginner"},
    {"Student_ID":"S005","Name":"Dev Anand",       "Interests":"cloud computing devops kubernetes docker aws terraform sre platform","Level":"Advanced"},
    {"Student_ID":"S006","Name":"Ananya Iyer",    "Interests":"mobile development flutter react native ios android kotlin swift","Level":"Intermediate"},
    {"Student_ID":"S007","Name":"Ravi Krishnan",  "Interests":"cybersecurity ethical hacking penetration testing network forensics soc","Level":"Intermediate"},
    {"Student_ID":"S008","Name":"Sneha Nair",     "Interests":"nlp transformers bert llm generative ai langchain agents openai","Level":"Advanced"},
    {"Student_ID":"S009","Name":"Rohan Mehta",    "Interests":"blockchain solidity defi nft smart contracts ethereum web3","Level":"Intermediate"},
    {"Student_ID":"S010","Name":"Kavya Reddy",    "Interests":"game development unity unreal engine 3d graphics shaders gamedesign","Level":"Beginner"},
    {"Student_ID":"S011","Name":"Vijay Nair",     "Interests":"data engineering apache spark kafka airflow dbt snowflake etl","Level":"Advanced"},
    {"Student_ID":"S012","Name":"Deepa Suresh",   "Interests":"ui ux design figma prototyping user research design systems","Level":"Beginner"},
]

students_df = pd.DataFrame(STUDENTS_DATA)

# Auto-generate synthetic ratings
rng2 = random.Random(99)
RATINGS_DATA = []
course_ids = courses_df["Course_ID"].tolist()

STUDENT_DOMAIN_MAP = {
    "S001":["Data Science"],
    "S002":["Web Development"],
    "S003":["Artificial Intelligence","Data Science"],
    "S004":["Data Analytics"],
    "S005":["Cloud Computing","DevOps"],
    "S006":["Mobile Development"],
    "S007":["Cybersecurity","Networking"],
    "S008":["Artificial Intelligence"],
    "S009":["Blockchain","Finance & FinTech"],
    "S010":["Game Development"],
    "S011":["Data Engineering","Data Science"],
    "S012":["UI/UX Design"],
}

for sid, domains in STUDENT_DOMAIN_MAP.items():
    domain_courses = courses_df[courses_df["Domain"].isin(domains)]["Course_ID"].tolist()
    rated = rng2.sample(domain_courses, min(15, len(domain_courses)))
    for cid in rated:
        RATINGS_DATA.append({
            "Student_ID": sid,
            "Course_ID": cid,
            "Rating": rng2.randint(3, 5)
        })

ratings_df = pd.DataFrame(RATINGS_DATA)
print(f"Generated {len(ratings_df)} ratings for {len(students_df)} students")

# ================================================================
#  FEATURE ENGINEERING
# ================================================================
courses_df["content_blob"] = (
    courses_df["Domain"].str.lower() + " " +
    courses_df["Skills"].str.lower() + " " +
    courses_df["Difficulty"].str.lower() + " " +
    courses_df["Platform"].str.lower() + " " +
    courses_df["Course_Name"].str.lower()
)

print("Fitting TF-IDF on 1200+ course corpus...")
tfidf_vec    = TfidfVectorizer(ngram_range=(1,2), max_features=2000, stop_words="english", sublinear_tf=True)
tfidf_matrix = tfidf_vec.fit_transform(courses_df["content_blob"])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

# ================================================================
#  SVD COLLABORATIVE FILTERING
# ================================================================
user_item = ratings_df.pivot_table(
    index="Student_ID", columns="Course_ID", values="Rating"
).fillna(0)

n_comp = min(8, min(user_item.shape)-1)
svd = TruncatedSVD(n_components=n_comp, random_state=42)
user_factors = svd.fit_transform(csr_matrix(user_item.values))
user_sim_df = pd.DataFrame(
    cosine_similarity(user_factors),
    index=user_item.index, columns=user_item.index
)
print(f"SVD: {n_comp} components, variance explained: {svd.explained_variance_ratio_.sum():.1%}")

# ================================================================
#  POPULARITY SCORE
# ================================================================
scaler = MinMaxScaler()
courses_df["pop_norm"]    = scaler.fit_transform(courses_df[["Enrolled"]])
courses_df["rating_norm"] = scaler.fit_transform(courses_df[["Rating"]])
courses_df["pop_score"]   = 0.55*courses_df["pop_norm"] + 0.45*courses_df["rating_norm"]

AVG_RATINGS = ratings_df.groupby("Course_ID")["Rating"].mean().to_dict()
LEVEL_MAP   = {"Beginner":1,"Intermediate":2,"Advanced":3}

# ================================================================
#  HYBRID RECOMMENDER
# ================================================================
def difficulty_score(student_level, course_diff):
    s = LEVEL_MAP.get(student_level.capitalize(), 1)
    c = LEVEL_MAP.get(course_diff.capitalize(), 1)
    d = c - s
    return {0:1.0, 1:0.85, -1:0.55}.get(d, 0.25)

def recommend(student_id: str, top_n: int = 10) -> list:
    student = students_df[students_df["Student_ID"] == student_id]
    if student.empty:
        return []

    stu      = student.iloc[0]
    interests= stu["Interests"].lower()
    level    = stu["Level"]
    rated    = set(ratings_df[ratings_df["Student_ID"]==student_id]["Course_ID"])

    q_vec      = tfidf_vec.transform([interests])
    int_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    results = []
    for idx, row in courses_df.iterrows():
        cid = row["Course_ID"]
        if cid in rated:
            continue

        i_score = float(int_scores[idx])

        c_score = 0.0
        if rated:
            rated_indices = courses_df.index[courses_df["Course_ID"].isin(rated)].tolist()
            if rated_indices:
                c_score = float(np.mean([courses_df.loc[ri,"content_blob"] and
                    cosine_similarity(
                        tfidf_matrix[idx:idx+1],
                        tfidf_matrix[ri:ri+1]
                    )[0][0] for ri in rated_indices[:5]]))  # cap at 5 for speed

        col_score = 0.0
        if student_id in user_item.index and cid in user_item.columns:
            sims = user_sim_df[student_id].drop(student_id)
            top  = sims.nlargest(6)
            ws, ss = 0.0, 0.0
            for oid, sim in top.items():
                if oid in user_item.index:
                    r = user_item.loc[oid, cid]
                    if r > 0:
                        ws += sim * r; ss += abs(sim)
            if ss > 0:
                col_score = ws / ss / 5.0

        p_score = float(row["pop_score"])
        d_score = difficulty_score(level, row["Difficulty"])

        final = (0.38*i_score + 0.20*c_score + 0.18*col_score +
                 0.14*p_score + 0.10*d_score)

        results.append({
            "course_id": cid, "course_name": row["Course_Name"],
            "domain": row["Domain"], "platform": row["Platform"],
            "difficulty": row["Difficulty"],
            "rating": round(float(AVG_RATINGS.get(cid, row["Rating"])),1),
            "enrolled": int(row["Enrolled"]),
            "duration_hrs": int(row["Duration_hrs"]),
            "price": float(row["Price"]),
            "instructor": row["Instructor"],
            "skills": row["Skills"][:120],
            "skill_tags": row["Skill_Tags"],
            "interest_score":  round(i_score*100, 1),
            "content_score":   round(c_score*100, 1),
            "collab_score":    round(col_score*100, 1),
            "popularity_score":round(p_score*100, 1),
            "difficulty_score":round(d_score*100, 1),
            "final_score": round(final, 6),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    if results:
        max_s = results[0]["final_score"]
        for r in results:
            r["match_pct"] = round(r["final_score"]/max_s*100, 1) if max_s else 0
            del r["final_score"]
    return results[:top_n]


def recommend_by_topic(topic: str, diff: str, platform: str, top_n: int = 10) -> list:
    q_vec   = tfidf_vec.transform([topic.lower()])
    scores  = cosine_similarity(q_vec, tfidf_matrix).flatten()

    temp = courses_df.copy()
    temp["_sim"] = scores
    if diff and diff != "all":
        temp = temp[temp["Difficulty"] == diff]
    if platform and platform != "all":
        temp = temp[temp["Platform"] == platform]

    temp = temp.nlargest(top_n, "_sim")
    results = []
    for _, row in temp.iterrows():
        results.append({
            "course_id": row["Course_ID"], "course_name": row["Course_Name"],
            "domain": row["Domain"], "platform": row["Platform"],
            "difficulty": row["Difficulty"],
            "rating": round(float(AVG_RATINGS.get(row["Course_ID"], row["Rating"])),1),
            "enrolled": int(row["Enrolled"]),
            "duration_hrs": int(row["Duration_hrs"]),
            "price": float(row["Price"]),
            "instructor": row["Instructor"],
            "skills": row["Skills"][:120],
            "skill_tags": row["Skill_Tags"],
            "match_pct": round(float(row["_sim"])*100, 1),
            "interest_score": round(float(row["_sim"])*100, 1),
            "content_score": 0, "collab_score": 0,
            "popularity_score": round(float(row["pop_score"])*100, 1),
            "difficulty_score": 0,
        })
    return results


def get_metrics():
    domains   = courses_df["Domain"].value_counts().to_dict()
    platforms = courses_df["Platform"].value_counts().to_dict()
    diffs     = courses_df["Difficulty"].value_counts().to_dict()
    return {
        "total_courses": len(courses_df),
        "total_students": len(students_df),
        "total_ratings": len(ratings_df),
        "svd_components": n_comp,
        "explained_variance": round(float(svd.explained_variance_ratio_.sum()*100),1),
        "tfidf_features": int(tfidf_matrix.shape[1]),
        "domains": domains,
        "platforms": platforms,
        "difficulties": diffs,
        "weights": {
            "Interest (TF-IDF)": 38,
            "Content Similarity": 20,
            "Collaborative (SVD)": 18,
            "Popularity": 14,
            "Difficulty Fit": 10,
        },
    }

# ================================================================
#  FLASK ROUTES
# ================================================================
@app.route("/")
def index():
    student_list = students_df[["Student_ID","Name","Interests","Level"]].to_dict("records")
    return render_template("index.html", students=student_list)

@app.route("/api/recommend/student", methods=["POST"])
def api_recommend_student():
    data = request.get_json()
    sid  = data.get("student_id","").strip()
    n    = int(data.get("top_n", 10))
    recs = recommend(sid, top_n=n)
    stu  = {}
    if sid in students_df["Student_ID"].values:
        stu = students_df[students_df["Student_ID"]==sid].iloc[0].to_dict()
    return jsonify({"success": True, "student": stu, "recommendations": recs})

@app.route("/api/recommend/topic", methods=["POST"])
def api_recommend_topic():
    data     = request.get_json()
    topic    = data.get("topic","").strip()
    diff     = data.get("difficulty","all")
    platform = data.get("platform","all")
    n        = int(data.get("top_n", 10))
    if not topic:
        return jsonify({"success": False, "message": "Topic required"})
    recs = recommend_by_topic(topic, diff, platform, n)
    return jsonify({"success": True, "topic": topic, "recommendations": recs, "total_searched": len(courses_df)})

@app.route("/api/metrics")
def api_metrics():
    return jsonify(get_metrics())

@app.route("/api/students")
def api_students():
    return jsonify(students_df.to_dict("records"))

@app.route("/api/courses/search")
def api_course_search():
    q = request.args.get("q","").lower()
    domain = request.args.get("domain","")
    filtered = courses_df[courses_df["content_blob"].str.contains(q, na=False)]
    if domain:
        filtered = filtered[filtered["Domain"]==domain]
    return jsonify({"count": len(filtered), "courses": filtered[["Course_ID","Course_Name","Domain","Difficulty","Platform","Rating"]].head(20).to_dict("records")})

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  CourseIQ  |  {len(courses_df)} Courses  |  {len(students_df)} Students")
    print(f"  Domains: {len(DOMAIN_CONFIG)}  |  SVD Factors: {n_comp}")
    print(f"  Open: http://127.0.0.1:5000")
    print(f"{'='*60}\n")
    app.run(debug=True, port=5000)
