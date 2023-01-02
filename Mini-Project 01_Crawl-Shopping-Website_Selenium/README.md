# Mini-Project 01 Crawl the shopping website using 'selenium'
Your task is to retrieve **the top-5 most expensive products** from our [CDS Shop](http://10.113.178.219) (UM VPN is required if you are not using the campus network).
The website is **not a static html**, you are encouraged to crawl the website using 'selenium'.
### You can follow these procedures to complete this task.
1. Search the empty keyword(keyword = '') to get every products
2. fetch the pages with given keyword
3. There're serval products on one page. To get the product detail, it requires you to access the product page by the 'href url'
4. get all the links of products on that page.
5. Use a loop to get all the product pages.
6. Parse the product name and price from that page.
7. Store the product information
8. Find the next_page url
9. If next_page url exist then fetch the next_page and go back to Step (2)
10. Sort the products by price and retrieve the names of top 5 as the result
11. Save the top5 result as json file, name the file "result.json".
### Hints: 
1. It's recommended to use a 'dict' to store the retrieving result (products name as key and price as value).
2. All the above processes are run on your own local machine (jupyterhub is not working).
3. A template code to parse the html pages is given.