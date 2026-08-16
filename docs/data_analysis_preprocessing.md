## Data Lineage & Phase-Wise Dataset Usage

| Phase | Input Dataset / File | Script / Module | Output / Destination Artifact |
| :--- | :--- | :--- | :--- |
| **1. Data Preprocessing** | `data/raw/Training_data.csv`<br>`data/raw/Query_and_Validation_data.csv` | `src.product_classifier.data.preprocess` | Cleaned train/validation datasets in `data/processed/` |

## Data Issue

### Training_data.csv

1. **Extra comma between name**

   - **False:**  
     `31: frozen meals Three Cheese Ziti, Marinara with Meatballs,Fresh & Perishable Items;;;;`

   - **True:**  
     `42571: frozen meals Three Cheese Ziti Marinara,Fresh & Perishable Items;;;;`

2. **Mixing product type in the front part of the description with lower classifier**

   - **False:**  
     `3: spices seasonings All-Seasons Salt,Dry Goods & Pantry Staples;;;;`

   - **True:**  
     `4: Robust Golden Unsweetened Oolong Tea,Beverages;;;;`

3. **Trailing `;;;;` affect the standardized processing pipeline**

   - **False:**  
     `2: cookies cakes Chocolate Sandwich Cookies,Dry Goods & Pantry Staples;;;;`

4. **Leading comma causes processing break**

   - **False:**  
     `42739: ,"Band-Aid Medium Hurt-Free Wrap 2\"",Household & Personal Care"`

   - **True:**  
     `42740: deodorants Cedarwood Juniper Deodorant,Household & Personal Care`

---

### Query_and_Validation_data.csv

1. **Descriptions with commas**

   - **False:**  
     `4: Cheesecake, Chocolate Truffle,`

   - **True:**  
     `5: White Multifold Towels,Household & Personal Care`

2. **Invalid character in description**

   - **False:**  
     `290: "bakery desserts 4\"" Banana Cream Pie,"`

   - **True:**  
     `303: Mini Falafel With Tahini Sauce,Fresh & Perishable Items`

---

## Data Processing Logic

1. Remove the first row (header) and `;;;;`.
2. Keep trailing commas and parse the rows.
3. Assign the columns as `product_name` and `category`.
4. Check if the row ends with a comma:
   - **Yes:** entire row is `product_name`, category is empty.
   - **No:** last value is `category`, remaining values are `product_name`.
5. Clean malformed quotes in product descriptions.
6. Remove duplicate rows.
7. Save the cleaned data.

---

## Notes

* **Expected Category Whitelist:**
  - `Fresh & Perishable Items`
  - `Beverages`
  - `Dry Goods & Pantry Staples`
  - `Household & Personal Care`
  - `Specialty & Miscellaneous`

* **Extra Lowercase Product Type Initial:** 
  - `Keep for text classification, no processing on initial and case handling.`