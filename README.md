# **E-commerce Customer Churn Prediction using Machine Learning**

## **Background Business Problem Understanding**

<img src ='https://www.zonkafeedback.com/hs-fs/hubfs/leading-causes-of-customer-churn-1.png?width=1920&name=leading-causes-of-customer-churn-1.png'>

Customer churn, also known as customer attrition, is a significant concern for e-commerce businesses. It refers to the phenomenon where customers stop doing business with an e-commerce platform. Churn can occur due to various reasons, such as dissatisfaction with the product or service, better offers from competitors, or changes in the customer’s situation or needs.

Let’s consider a real-world example of an e-commerce business facing customer churn. Suppose an online shop started with 700 customers in the month of September and ended with 650. The churn rate for this business would be about 7%. This rate is significant because acquiring new customers can cost five times more than retaining existing ones

An e-commerce company that operates in the online buying and selling business is one of the solutions in the digital world that makes it easier for customers to meet their needs without having to visit the seller’s store directly. Online e-commerce offers many conveniences for customers, including the ability to compare many products at once to get a cheap and quality product, a wide variety of goods available in one display, providing various payment methods from cash payments to installments through credit cards or financial partners, and there are many promotions from discounts to cashback offered to customers. In the online e-commerce business, customers are an important component of the company’s revenue, for example, customers can provide direct income when the customer makes a transaction through the company’s online platform, or the level of crowdedness or customer traffic on the company’s online platform can also attract brands to collaborate to market their products through the company’s online platform. 

Therefore, it is very important for the company to be able to maintain its customers to remain loyal and not move to other company’s online platforms (Customer Churn). We can see in the case of Netflix, which is an online platform for watching movies, in the first quarter of 2022, Netflix lost up to 200,000 customers, which can certainly have a significant impact on the company’s revenue.

Customer Churn, also known as customer migration, is the termination of a company’s services by customers because the customer chooses to use other services. By predicting churn, the company can identify churn earlier so that the company’s losses due to migrating consumers can be avoided. Consumers are the main assets of the company, so one way for the company to retain consumers is by recognizing potential customers and must be able to retain potential customers (customer retention) so that it can prevent customers from stopping purchases and moving to competitor companies (churn).

According to a study by Neil T. Awit and Ramon M. Marticio, acquiring new customers costs five times the cost of satisfying and retaining old customers, while a 2% increase in retaining customers (customer retention) has an impact on profits such as cutting costs by 10%. By applying churn prediction, the company can identify customer churn and apply the right marketing strategy to old customers in the hope of increasing the company’s revenue.

To manage customer churn effectively, businesses need to implement strategies such as customer segmentation and churn modeling. Customer segmentation involves dividing the customer base into smaller, more cohesive groups of individuals that are similar in specific ways. These customer groups can then be targeted with personalized marketing campaigns, loyalty programs, or other retention strategies

**Target:**

**- 0 : Customers continue to use E-commerce/Loyal**

**- 1 : Customers leave using E-commerce/Exit**

References:

https://books.google.com/books/about/Leading_on_the_Edge_of_Chaos.html?id=LOPtAAAAMAAJ

https://iucat.iu.edu/iupui/5168290

https://article.sciencepublishinggroup.com/pdf/10.11648.j.ajtas.20190806.18.pdf

https://www.researchgate.net/profile/Neil-Awit/publication/371738059_Customer_Churn_Prediction_using_Predictive_Analytics_Basis_for_the_Formulation_of_Customer_Retention_Strategy_in_the_Context_of_Web-based_Collaboration_Platform/links/649927638de7ed28ba58fab2/Customer-Churn-Prediction-using-Predictive-Analytics-Basis-for-the-Formulation-of-Customer-Retention-Strategy-in-the-Context-of-Web-based-Collaboration-Platform.pdf?origin=publication_detail

http://www.sciencepublishinggroup.com/j/ajtas

https://whiplash.com/blog/causes-of-customer-churn/

### Problem Statement

One of the significant challenges in online business is how a company can retain its customers and prevent them from switching to other platforms. Customers are a company’s primary asset, and one way a company can retain them is by predicting customer churn. By predicting churn, a company can identify potential customers for retention and implement appropriate marketing strategies, such as offering discounts or cashback to customers who are likely to churn, thereby preventing these customers from discontinuing purchases and switching to competitors.

The existence of a customer churn prediction model allows a company to minimize losses due to the loss of a number of customers because the company can identify loyal and non-loyal customers. Thus, the cost incurred to attract new customers can be avoided by retaining loyal customers, where the cost to retain existing customers is relatively lothis compared to attracting new customers.

To elaborate further, customer churn prediction is a process that identifies customers who are at a high risk of canceling their subscription or abandoning this product. A churn prediction model is a machine learning model that predicts whether a customer will likely churn. This model works by passing previous customer data through a machine learning model to identify the connections between features and targets and make predictions about new customers.

The first step to creating a churn model is to collect relevant data, including product usage data and direct feedback data from customer surveys. Next, we’ll need to analyze trends in the data to find the main reasons behind customer churn. Finally, we’ll pass the data through a logistic regression algorithm (such as the random forest algorithm) to identify key data points and make future predictions.

Understanding and predicting customer churn is crucial because the cost of acquiring new customers is always higher than the cost of retaining existing ones. Therefore, lothising churn has a big positive impact on this revenue streams. Churn rates track lost customers, and growth rates track new customers—comparing and analyzing both of these metrics tells we exactly how much this business is growing over time. If growth is higher than churn, we can say this business is growing. If churn is higher than growth, this business is getting smaller.

## **Goals**
---
<img src = 'https://a.storyblok.com/f/105861/1226x697/78578ca5a5/churn-prediction-training-revised-text-v2.png'>

**Based on the problems faced, the company must be able to predict which customers are likely to churn and then provide the appropriate treatment to these customers to prevent churn. This way, the company can maintain the profits it has earned. Also, it can minimize the retention cost needed for customers who are about to churn.**


> ## **Analytic Approach**
---

We will analyze the data to find patterns that distinguish customers who will churn from those who will not. Then, we will build a classification model that will assist the company in predicting whether a customer will churn or not.


### **Confusion Matrix Terms:**

![Confussion Matrix](https://www.nbshare.io/static/snapshots/cm_colored_1-min.png)

|       | N-Pred| P-Pred |
| --- | --- | --- |
| **N-Act**     | TP | FP |
| **P-Act**      | FN | TP |

1. TP: The customer actually churns and is predicted to churn

2. TN: The customer actually does not churn and is predicted not to churn

3. FN: The customer actually churns but is predicted not to churn

4. FP: The customer actually does not churn but is predicted to churn

**Cost of FN (False Negative):**

**1. Disadvantages**

- Loss of customers (i.e., churn)

- The cost of customer acquisition to replace customers who have churned

- Condition and Impact: This is a situation where the model predicts a customer will not churn, but in reality, the customer churns. A prediction failure in this condition causes the company to lose customers without knowing the reason and without any anticipatory steps to retain those customers. With the sudden loss of a number of these customers, there are two very important losses for the company: the company loses direct revenue from customers, and the company loses the opportunity to map out improvement steps to prevent customers from churning.

**Cost of FP (False Positive):**

**1. Advantages**

- As a result of incorrect treatment of customers who actually do not churn but are predicted to churn, the reputation of the E-Commerce platform improves (customers who do not churn will think that the E-Commerce platform is generous in giving promotions for free) 

**2. Disadvantages**

- Incorrect target treatment for customers who do not churn (but are predicted to churn)

- Wasted customer retention costs, time, and resources

- Condition and Impact: This is a situation where the model predicts a customer will churn, but in reality, the customer does not churn. A prediction failure in this condition will not cause the company to lose customers because the incorrectly predicted customers will continue to use the company’s online platform. The impact of the company’s steps in anticipating customer churn, such as giving promotions to customers, could potentially increase the customer’s transaction activity, and the characteristics of the customers can still be used for service performance improvement.

Based on these consequences, as much as possible, we will create a model that can reduce the customer retention cost of the company without having to have customers churn from the company’s E-Commerce website. Therefore, we decide to focus on False Negatives, but also not forget about False Positives, with more emphasis on recall. Hence, the focus metric we use is the F2-Score.
The F2-Score is a measure of a test’s accuracy that considers both precision (the number of correct positive results divided by the number of all positive results) and recall (the number of correct positive results divided by the number of positive results that should have been returned). The F2-Score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 (perfect precision and recall) and its worst value at 0.

**In the F2-Score, recall is considered twice as important as precision. This means that it’s more important to capture as many actual positives as possible, even if it means incorrectly predicting some negatives as positives. This makes the F2-Score particularly useful in situations like ours, where the cost of False Negatives (losing a customer) is higher than the cost of False Positives (unnecessary promotions).**

References:

https://deepchecks.com/glossary/f-score/

https://permetrics.readthedocs.io/en/latest/pages/classification/F2S.html

https://en.wikipedia.org/wiki/F-score

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html

## **Customer Retention Cost Analysis**

<img src = 'https://miro.medium.com/v2/resize:fit:1358/1*63IjZR8hQ6hA5Qp0GnCRMQ.png'>

**Cost of Retaining a Customer:**
- **Cost:** $10 per customer to keep them happy and shopping with the company (this could be through good customer service, discounts, etc.).

**Profit from a Customer:**
- **Profit:** Each customer brings in $100 in profit.

**Net Profit from a Customer:**
- After spending $10 to keep the customer, the company makes $90 in profit.
  - Profit: $100
  - Retention Cost: $10
  - Net Profit: $100 - $10 = $90

**Cost of Acquiring a New Customer:**
- **Cost:** $50 to acquire a new customer.

**Lost Profit due to Churn:**
- If a customer churns, the company loses both the $50 spent to acquire a new customer and the $90 they would have made if the old customer had stayed.
  - Lost Profit: $50 + $90 = $140

**5X Loss for the company:**
- The $140 loss due to churn is more than 5 times the cost of retaining a customer ($10).
  - Loss: $140
  - Retention Cost: $10
  - Ratio: $140 / $10 = 14 times more costly to acquire a new customer than to retain an existing one.

In conclusion, customer retention is crucial for a company’s profitability. The cost of acquiring a new customer can be significantly higher than the cost of retaining an existing one. In our example, the company loses more than 5 times the cost of retaining a customer when a customer churns and a new one is acquired. This is due to the combined loss of the profit that would have been made from the churned customer and the cost of acquiring a new customer. Therefore, investing in customer retention strategies can be a more cost-effective way for companies to maintain their profitability. Remember, these are simplified numbers and the actual costs can vary a lot depending on the business. But the principle remains the same: keeping customers is usually cheaper than finding new ones.
  

## **Columns Description**

| Attribute Name         | Description                                                   | Type          |
|------------------------|---------------------------------------------------------------|---------------|
| Tenure                 | Tenure of a customer in the company                            | Numeric       |
| WarehouseToHome        | Distance between the warehouse and the customer’s home         | Numeric       |
| NumberOfDeviceRegistered| Total number of devices registered to a particular customer    | Numeric       |
| PreferredOrderCat      | Preferred order category of a customer in the last month       | Categorical   |
| SatisfactionScore      | Satisfaction score of a customer on service                    | Numeric       |
| MaritalStatus          | Marital status of a customer                                  | Categorical   |
| NumberOfAddress        | Total number of addresses added by a particular customer       | Numeric       |
| Complaint              | Whether any complaint has been raised by the customer          | Binary        |
| DaySinceLastOrder      | Number of days since the last order by the customer            | Numeric       |
| CashbackAmount         | Average cashback received by the customer in the last month    | Numeric       |
| Churn                  | Churn flag indicating whether the customer has churned         | Binary        |

## **Conclusion for E-commerce Company**

## **In term of Potential Lost and Deficit**
<img src = 'https://etimg.etb2bimg.com/thumb/msid-79736173,width-1200,height-900,resizemode-4/.jpg'>

| Scenario                          | Loss due to FN | Loss due to FP | Total Potential Loss |
|-----------------------------------|----------------|----------------|----------------------|
| Before Tuning                    | $2790          | $180           | $2970                |
| After Tuning                     | $450           | $910           | $1360                |
| After Tuning and Threshold Adjustment | $630      | $700           | $1330                |

**Scenario 1: Before Tuning**

- Loss due to FN: $90 * 31 = $2790
- Loss due to FP: $10 * 18 = $180
- Total potential loss: $2790 + $180 = $2970

**Scenario 2: After Tuning**

- Loss due to FN: $90 * 5 = $450
- Loss due to FP: $10 * 91 = $910
- Total potential loss: $450 + $910 = $1360

**Scenario 3: After Tuning and Threshold Adjustment**

- Loss due to FN: $90 * 7 = $630
- Loss due to FP: $10 * 70 = $700
- Total potential loss: $630 + $700 = $1330

**Based on these calculations, the potential loss is lowest in Scenario 3 (After Tuning and Threshold Adjustment), followed by Scenario 2 (After Tuning). The highest potential loss is in Scenario 1 (Before Tuning).**

**So, if we're prioritizing minimizing the potential loss due to false negatives and false positives, the company should opt for either Scenario 3 (After Tuning and Threshold Adjustment) or Scenario 2 (After Tuning).**


## **In term of Profit and Savings**
<img src ='https://www.xero.com/content/dam/xero/pilot-images/glossary/profit-calculation.1653201143123.png'>
To compare the three scenarios and determine which one is the most beneficial for the e-commerce company in terms of minimizing deficit in profit, let's analyze the net profit and potential savings for each scenario.

|                       | Before Tuning | After Tuning | After Tuning and Threshold Adjustment |
|-----------------------|---------------|--------------|--------------------------------------|
| Net Profit            | \$47,520      | \$45,040     | \$45,120                             |
| Potential Savings     | \$3,750       | \$5,050      | \$4,950                              |
| Total Profit          | \$51,270      | \$50,090     | \$50,070                             |

1. Scenario 1 (Before Tuning) has the highest total profit, but it also has the lowest potential savings. This scenario might result in higher net profit but could miss out on potential savings from reducing false negatives and false positives.

2. Scenario 2 (After Tuning) has the highest potential savings, which means the model is effectively reducing false negatives. However, the net profit is slightly lower compared to Scenario 1.

3. Scenario 3 (After Tuning and Threshold Adjustment) has a similar net profit to Scenario 2 but slightly lower potential savings. However, it still performs better than the initial model (Scenario 1).
Conclusion:

**Scenario 2 (After Tuning) appears to strike a better balance between net profit and potential savings. By reducing false negatives, the company can focus its retention strategies more effectively, leading to significant potential savings. Although the net profit is slightly lower compared to the initial model, the overall benefit of reducing false negatives outweighs this reduction.**

Recommendation:

Further tuning and optimization could be beneficial to reduce the number of false positives in Scenario 2, which would further improve the model's performance and potentially increase net profit. Additionally, continuous monitoring and updating of the model based on new data and changing business needs are essential to ensure its effectiveness over time.

**Therefore, Scenario 3 (After Tuning and Threshold) is recommended for the e-commerce company to minimize the deficit in profit while maximizing potential savings from effective churn prediction and retention strategies.**

## **A more brief Explanation in Term of Profit and Saving**

**Scenario 1: Before Tuning**

- Net Profit = Total Profit from TN - Total Cost for TN + Total Profit from TP - Total Cost for FP
- Net Profit = ($90 * 528) - ($10 * 528) + ($90 * 75) - ($10 * 18)
- Net Profit = $47,520 + $7,260
- Net Profit = $54,780

**Scenario 2: After Tuning**

- Net Profit = Total Profit from TN - Total Cost for TN + Total Profit from TP - Total Cost for FP
- Net Profit = ($90 * 455) - ($10 * 455) + ($90 * 101) - ($10 * 91)
- Net Profit = $40,950 + $810
- Net Profit = $41,760

**Scenario 3: After Tuning and Threshold Adjustment**

- Net Profit = Total Profit from TN - Total Cost for TN + Total Profit from TP - Total Cost for FP
- Net Profit = ($90 * 476) - ($10 * 476) + ($90 * 99) - ($10 * 70)
- Net Profit = $42,840 + $1,190
- Net Profit = $44,030

Now, let's calculate the potential savings for each scenario:

|                       | Potential Savings |
|-----------------------|-------------------|
| Before Tuning         | $3,750            |
| After Tuning          | $5,050            |
| After Tuning and Threshold Adjustment | $4,950 |

**Scenario 1: Before Tuning**

- Potential Savings = Total Potential Savings from TP
- Potential Savings = $50 * 75
- Potential Savings = $3,750

**Scenario 2: After Tuning**

- Potential Savings = Total Potential Savings from TP
- Potential Savings = $50 * 101
- Potential Savings = $5,050

**Scenario 3: After Tuning and Threshold Adjustment**

- Potential Savings = Total Potential Savings from TP
- Potential Savings = $50 * 99
- Potential Savings = $4,950


**Based on these calculations:**

- Scenario 1 (Before Tuning) has a total profit of $54,780 and potential savings of $3,750.
- Scenario 2 (After Tuning) has a total profit of $41,760 and potential savings of $5,050.
- Scenario 3 (After Tuning and Threshold Adjustment) has a total profit of $44,030 and potential savings of $4,950.

**Therefore, the recommendation remains the same: Scenario 3 (After Tuning and threshold) strikes the best balance between net profit and potential savings.**

## **In Term of Total Revenue Gain**
<img src ='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTc9-mosC2otARmo8isMjTM9A4ZDyNuQENa1w&usqp=CAU'>

| Scenario                              | Net Profit | Potential Savings | Total Revenue |
|---------------------------------------|------------|-------------------|---------------|
| Before Tuning                        | $54,780    | $3,750            | $58,530       |
| After Tuning                         | $41,760    | $5,050            | $46,810       |
| After Tuning and Threshold Adjustment| $44,030    | $4,950            | $48,980       |

To determine which model to use in terms of revenue, we need to analyze the potential impact of each model on the company's revenue. We'll consider the net profit and potential savings as indicators of the model's effectiveness in retaining customers and minimizing churn-related costs.

Here are the calculations for each scenario:

**Before Tuning:**

- Net Profit: $54,780
- Potential Savings: $3,750

**After Tuning:**

- Net Profit: $41,760
- Potential Savings: $5,050

**After Tuning and Threshold Adjustment:**

- Net Profit: $44,030
- Potential Savings: $4,950

Now, let's compute the total revenue for each scenario:

**Total Revenue = Net Profit + Potential Savings**

**Before Tuning:**

- Total Revenue = $54,780 + $3,750 = $58,530

**After Tuning:**

- Total Revenue = $41,760 + $5,050 = $46,810

**After Tuning and Threshold Adjustment:**

- Total Revenue = $44,030 + $4,950 = $48,980

**Based on these calculations, the total revenue is highest for Scenario 1: Before Tuning. However, it's important to note that this scenario also has the highest net profit, but the lowest potential savings.**

**While Scenario 3: After Tuning and threshold has a lower net profit compared to Scenario 1, its higher potential savings indicate that the model is more effective at identifying customers likely to churn. This could lead to increased revenue in the long run by retaining more customers.**

Therefore, if the goal is to maximize revenue and effectively manage churn-related costs, Scenario 2: After Tuning would be the preferred choice.

## **Recommendation for E-commerce Company**

1. Implement the Tuned Model: The churn prediction model after tuning showed the highest potential savings and a reasonable net profit. This indicates that the model has been optimized to strike a balance between correctly identifying churners and minimizing unnecessary retention costs.

2. Continue Monitoring and Fine-Tuning: While the tuned model showed promising results, it's essential for the company to continue monitoring its performance and make further adjustments as needed. This includes periodically retraining the model with new data and refining the threshold for churn prediction to optimize cost-effectiveness continually.

3. Focus on False Negatives: Given that false negatives (customers who churn but are predicted to stay) can lead to significant profit loss, the company should pay special attention to reducing this type of error. Strategies could include refining feature selection, improving model performance on minority classes, or exploring alternative machine learning algorithms.

4. Invest in Customer Retention: With a more accurate churn prediction model in place, the company can allocate resources more effectively towards retaining existing customers. This could involve implementing targeted retention strategies based on the model's predictions, such as personalized offers, loyalty programs, or proactive customer support.

5. Balance Retention Costs with Customer Acquisition: While retaining existing customers is crucial for long-term profitability, it's also essential to balance retention costs with customer acquisition costs. The company should continuously evaluate the cost-effectiveness of retention strategies compared to acquiring new customers, ensuring that resources are allocated optimally.

**In conclusion, leveraging machine learning for churn prediction can provide valuable insights for e-commerce companies to optimize their retention efforts and maximize profitability. By implementing and fine-tuning predictive models, monitoring performance, and investing in targeted retention strategies, companies can reduce churn rates, retain valuable customers, and ultimately drive sustainable growth and profitability.**

## **Suggestion for Model**

1. Feature Engineering: Continue refining the features used in the model to capture more relevant information about customer behavior and preferences.

2. Threshold Adjustment: Further optimize the threshold for classifying churn to strike the right balance between false positives and false negatives based on the company's objectives and resources.

3. Ensemble Methods: Explore ensemble methods such as stacking or boosting to improve model performance by combining multiple models.

4. Regularization: Implement regularization techniques to prevent overfitting and improve generalization performance.

5. Data Collection and Quality: Ensure consistent and high-quality data collection processes to maintain model performance over time.

6. Continuous Monitoring and Updating: Regularly monitor model performance and update it as new data becomes available to adapt to changing customer behavior and market dynamics.
## Authors

* **Hieremias Kevin Juwantoro** - *Initial work* - [https://github.com/kevinjuwantoro/E-commerce-Churn-Machine-Learning)
