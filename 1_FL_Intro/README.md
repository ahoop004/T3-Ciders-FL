# Introduction to Federated Learning

## Overview

**Teaching:** 15 min  
**Exercises:** 0 min

**Questions**
- What is Federated Learning?
- How does Federated Learning differ from classical machine learning?
- What are the key challenges that Federated Learning addresses?
- How does the Federated Learning process work?

**Objectives**
- Understanding the fundamental concepts of Federated Learning
- Recognizing the limitations of classical centralized machine learning
- Comprehending the five-step Federated Learning process
- Identifying real-world applications and use cases
## What is Federated Learning?

Federated Learning is a machine learning approach that enables model training across multiple decentralized clients (devices or organizations) holding local data, **without exchanging the raw data samples themselves**. This paradigm represents a fundamental shift from traditional centralized machine learning approaches. [1][2]


<center>
<img src="FD_learning/resources/1_what_is_fd_learning.jpg" width="400"/><br>
Figure from: https://www.linkedin.com/pulse/federated-learning-healthcare-part-1-saber-ghadakzadeh-md-msc-phd/
<center> 

### Who/what is a “client” in FL?

A **client** is any participant that holds data locally and can compute updates.
The main two settings for clients are commonly called **cross-device** and **cross-silo** federated learning. [2]

- **Cross-device FL:** millions of phones / IoT devices (each device is a client)
- **Cross-silo FL:** a small number of **organizations/corporations** (e.g., hospitals, banks, companies, data centers) where each institution is a client


### Key Principle

The core principle of Federated Learning can be summarized in one line:
- **Classical Machine Learning:** Move the data to the computation
- **Federated Learning:** Move the computation to the data

This approach enables machine learning in scenarios where data cannot be centralized due to privacy concerns, regulatory requirements, or practical constraints.



## Classical Machine Learning vs. Federated Learning

### Classical Machine Learning

In classical machine learning, the process typically follows these steps:

1. **Data Collection:** Data from multiple sources is collected and centralized on a single server or data center
2. **Model Training:** The machine learning model is trained on the centralized dataset
3. **Model Deployment:** The trained model is deployed to various devices or systems


<center>
<img src="FD_learning/resources/2_machine_learning.png" width="400"/><br>
Figure from: https://7wdata.be/big-data/building-the-machine-learning-infrastructure/
<center>


### Federated Learning Approach

Federated Learning changes the training workflow so data stays local and only model updates move:

1. **Model Distribution:** A global model is distributed to multiple devices or organizations/corporations (cross-device vs. cross-silo FL)
2. **Local Training:** Each participant trains the model on their local data
3. **Model Aggregation:** Only model updates (not raw data) are sent back to the central server
4. **Global Update:** The central server aggregates these updates to improve the global model

<center>
<img src="FD_learning/resources/3_FD_learning.ppm" width="400"/><br>
Figure from: https://www.researchgate.net/figure/The-framework-of-Federated-Learning-Graphical-illustration-of-the-working-principle-of_fig1_367191647
<center>


## Challenges of Classical Machine Learning



The centralized approach faces several significant limitations, especially when data is sensitive, regulated, or expensive to move. [2]

### Regulatory Constraints

Various data protection regulations can restrict pooling data in a single location and/or transferring it across organizations and borders:
- **GDPR (Europe):** international transfers must meet Chapter V conditions (see Article 44 “General principle for transfers”). [7]
- **CCPA/CPRA (California):** grants consumers rights and imposes obligations on covered businesses regarding collection/use/sharing of personal information. [8]
- **HIPAA (Healthcare):** sets standards for protecting “protected health information” (PHI) and limits uses/disclosures without authorization. [9]
- **Financial privacy rules (U.S.):** GLBA/FTC Privacy Rule governs “nonpublic personal information” and requires privacy notices and limits certain sharing. [10]

### Privacy Concerns

Even when data transfer is legally possible, users and organizations may prefer not to centralize raw data:
- **Device privacy expectations:** many people report limited control over what companies and government do with their data, increasing pressure to minimize data movement. [11]
- **Organizational privacy:** corporations/institutions may be unable to share proprietary, sensitive, or contract-restricted data (a common motivation for cross-silo FL). [2]
- **End-to-end encryption:** E2EE systems are designed so content remains confidential between endpoints (not decryptable by intermediate servers), reinforcing expectations that sensitive content should not be exposed centrally. [12]

### Practical Limitations

Technical and economic constraints:
- **Bandwidth limitations:** communication is often the primary bottleneck at scale; transmitting large datasets to a central server can be slow and expensive. [1][2]
- **Storage costs & risk concentration:** central storage increases operational overhead and concentrates risk in a single high-value target. [2]
- **Real-time requirements:** some applications benefit from on-device/local processing and updates close to where data is generated. [2]



## The Federated Learning Process

Federated learning operates through a systematic five-step process that repeats until the model reaches convergence:

### Step 0: Initialize Global Model

The process begins with a central server initializing the global model. The model parameters are either set randomly or loaded from a pre-trained checkpoint, similar to classical machine learning approaches. This global model serves as the starting point for all participants in the federation.

### Step 1: Distribute Model to Client Nodes

Once initialized, the global model is distributed to participating clients (devices or organizations). In many deployments, only a subset of clients participates in a given training round. [2]

### Step 2: Local Training on Client Data

Each selected participant then performs local training using their own dataset. This local training is typically limited to a small number of epochs or mini-batches, as full convergence is not required at this stage. As a result, the models on individual clients develop slightly different parameters based on the unique characteristics of their local data.

### Step 3: Return Model Updates

After local training, each participant sends their model updates back to the central server. These updates generally consist of modified model parameters or gradients, ensuring that the raw training data remains on the local devices and is never transmitted. The collected updates reflect the learning that has occurred on the local datasets.

### Step 4: Aggregate Model Updates

Finally, the central server aggregates the model updates received from clients and uses them to **update** the global model. A widely used method is **Federated Averaging (FedAvg)**, which averages client updates (often weighted by how much local data each client used). [1]


### Step 5: Iterate Until Convergence

Steps 1 through 4 together make up a single round of federated learning. This process is then repeated, with each new round beginning by distributing the most recently updated global model to the selected participants. Multiple rounds are typically required for the model to fully converge and reach optimal performance. With each iteration, the global model gradually improves as it incorporates knowledge from all participating data sources, leading to better overall accuracy and generalizability.

## Real-World Applications

Federated learning is increasingly being adopted across a range of industries. In all of these examples, clients are collaboratively training a **shared single model** while keeping raw data local. [2]


### Healthcare

In the healthcare sector, federated learning enables hospitals to collaborate on medical image analysis for tasks such as disease detection, without the need to share sensitive patient data. By aggregating insights from larger and more diverse datasets spread across institutions, hospitals can achieve improved diagnostic accuracy while adhering to strict regulations surrounding medical privacy. Additionally, federated learning is being used in drug discovery, where pharmaceutical companies can work together on developing new treatments. This approach allows for the sharing of learning outcomes while protecting proprietary research data, thereby accelerating the discovery of effective therapies.[4]



<center>
<img src="FD_learning/resources/4_medical_detection.jpg" width="400"/><br>
Figure from: https://www.linkedin.com/pulse/federated-learning-healthcare-part-1-saber-ghadakzadeh-md-msc-phd/
<center>


### Financial Services

Financial institutions are leveraging federated learning for critical applications such as fraud detection. By collaborating on the development of fraud detection models, banks can benefit from a wider variety of financial data, significantly improving accuracy while keeping sensitive customer information protected. Another key area is risk assessment, where credit scoring models are trained across multiple institutions. This enables more accurate risk prediction without requiring the direct sharing of data, ensuring compliance with financial regulations and maintaining customer confidentiality.[6]


<center>
<img src="FD_learning/resources/5_FD_bank.webp" width="400"/><br>
Figure from: https://link.springer.com/article/10.1007/s00521-023-09410-2/figures/2
<center>


### Technology and IoT

In technology and IoT, federated learning has become prominent in the development of mobile applications. For example, smartphone keyboard prediction models are trained on user devices, enabling personalized recommendations and improved user experiences, all while ensuring that personal data never leaves the device. Federated learning is also vital in autonomous vehicles, where car manufacturers share learning from driving experiences. This not only improves safety and allows for real-time adaptation to local driving conditions, but it also protects proprietary sensor data from exposure.[5][2]


<center>
<img src="FD_learning/resources/7_FD_iot.ppm" width="400"/><br>
Figure from: https://www.researchgate.net/publication/356249953/figure/fig1/AS:1092587159076864@1637504486141/Federated-Learning-for-IoT-Devices.ppm
<center>

<center>
<img src="FD_learning/resources/6_FD_auto.png" width="400"/><br>
Figure from: https://www.semanticscholar.org/paper/Federated-Semi-Supervised-Learning-for-Object-in-Chi-Wang/569aafb945854b09ba3a47fc6376d83cced03597
<center>
## Advantages of Federated Learning

Federated learning offers several significant advantages that have contributed to its growing adoption across various sectors. 

### Privacy Preservation

A key benefit of federated learning is its ability to preserve privacy. Since data never leaves its original location, raw information remains on local devices or servers, and only model updates—rather than the sensitive data itself—are shared with the central system. This approach maintains data sovereignty and provides organizations with greater control over their information. Furthermore, federated learning supports regulatory compliance by meeting strict data protection requirements that are common in many industries. It enables secure collaboration across different regions or countries without violating cross-border data regulations and is highly adaptable to sector-specific privacy mandates.[2][3]

### Scalability and Efficiency

In terms of scalability and efficiency, federated learning leverages distributed processing, utilizing computing resources from multiple locations. This helps to ease the computational burden on any single central server, allowing for more efficient model training and even enabling real-time local processing where needed. Additionally, by transmitting only the necessary model parameters instead of the entire dataset, federated learning significantly reduces the amount of data transferred across the network. This optimization makes the approach highly suitable for environments with limited bandwidth, ensuring effective collaboration without overwhelming network resources.
## Challenges and Limitations

Federated learning, while promising, also faces several challenges and limitations in its practical use.

### Technical Challenges

One notable challenge is the communication overhead involved. Federated learning requires many rounds of sending models back and forth between the central server and participants, which can lead to increased network latency and reliability problems. Keeping all participants synchronized throughout the training process can be difficult, especially when network conditions are less than ideal. Additionally, model convergence is typically slower than in centralized learning. This is partly because the data is often distributed unevenly—different participants might have very different types of data—which makes it harder for the collective model to learn efficiently. When the data across participants is not independent and identically distributed (non-IID), training can be even more complicated and less stable.

### Security Considerations

Security is another important concern in federated learning. Although the method is designed to protect raw data, it can still be vulnerable to privacy attacks, such as model inversion or membership inference, where an attacker tries to extract sensitive information from shared model updates. Therefore, extra privacy-preserving techniques, like differential privacy or secure aggregation, are often needed. Moreover, ensuring trust and verification among all participants can be challenging. It’s important to detect any malicious users who might provide false or harmful model updates, and to have methods in place for verifying that each contribution to the global model is legitimate.

## Next Steps

This introduction provides the foundation for understanding Federated Learning. In subsequent modules, we will explore:

- **Practical Implementation:** Building federated learning systems
- **Advanced Algorithms:** Sophisticated aggregation and optimization techniques
- **Security and Privacy:** Advanced protection mechanisms
- **Real-World Case Studies:** Successful deployments and applications

Federated Learning represents not just a technical innovation, but a fundamental rethinking of how we approach collaborative AI development in a privacy-conscious world.

## References

[1] McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (AISTATS 2017). https://proceedings.mlr.press/v54/mcmahan17a.html  
[2] Kairouz et al. *Advances and Open Problems in Federated Learning* (arXiv:1912.04977). https://arxiv.org/abs/1912.04977  
[3] Bonawitz et al. *Practical Secure Aggregation for Privacy-Preserving Machine Learning* (ePrint 2017/281). https://eprint.iacr.org/2017/281  
[4] Rieke et al. *The future of digital health with federated learning* (npj Digital Medicine, 2020). https://doi.org/10.1038/s41746-020-00323-1  
[5] Hard et al. *Federated Learning for Mobile Keyboard Prediction* (arXiv:1811.03604). https://arxiv.org/abs/1811.03604  
[6] Zheng et al. *Federated Meta-Learning for Fraudulent Credit Card Detection* (IJCAI 2020). https://doi.org/10.24963/ijcai.2020/642  
[7] GDPR (EUR-Lex). *Regulation (EU) 2016/679* (GDPR) — Article 44 “General principle for transfers”. https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng  
[8] California DOJ. *California Consumer Privacy Act (CCPA)* overview. https://oag.ca.gov/privacy/ccpa  
[9] U.S. HHS. *The HIPAA Privacy Rule* overview. https://www.hhs.gov/hipaa/for-professionals/privacy/index.html  
[10] U.S. FTC. *How to Comply with the Privacy of Consumer Financial Information Rule (GLBA)*. https://www.ftc.gov/tips-advice/business-center/guidance/how-comply-privacy-consumer-financial-information-rule-gramm  
[11] Pew Research Center. *Americans and Privacy: Concerned, Confused and Feeling Lack of Control Over Their Personal Information* (2019). https://www.pewresearch.org/internet/2019/11/15/americans-and-privacy-concerned-confused-and-feeling-lack-of-control-over-their-personal-information/  
[12] IETF Internet-Draft. *Definition of End-to-end Encryption* (draft-knodel-e2ee-definition). https://datatracker.ietf.org/doc/html/draft-knodel-e2ee-definition-11  

