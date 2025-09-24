# Databricks notebook source
# MAGIC %pip install databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

from databricks.sdk.service.marketplace import ConsumerTerms

def get_current_user()-> str:
  return (dbutils.notebook.entry_point
    .getDbutils()
    .notebook()
    .getContext()
    .userName()
    .get()
    .split("@")[0]
    .replace(".", "_"))

def get_terms_of_service_version(terms_of_service: str)->str:
  return terms_of_service.split("/files/")[1].split("/")[0]

def install_marketplace_tables(listing, catalog_suffix:str) -> None:
  consumer_terms_version = get_terms_of_service_version(listing.detail.terms_of_service)
  w.consumer_installations.create(
    listing_id=listing.id, 
    share_name=listing.summary.share.name, 
    catalog_name=f"{get_current_user()}_{catalog_suffix}",
    accepted_consumer_terms=ConsumerTerms(version=consumer_terms_version)
  )

def delete_marketplace_tables(listing_id:str, catalog_suffix:str) -> None:
  installation = [
      installation
      for installation in w.consumer_installations.list()
      if installation.catalog_name == f"{get_current_user()}_{catalog_suffix}"
  ][0]

  w.consumer_installations.delete(
      installation_id=installation.id,
      listing_id=listing_id,
  )

# COMMAND ----------

simulated_retail_listing =  [marketplace_list for marketplace_list in w.consumer_listings.list() if marketplace_list.summary.name=='Simulated Retail Customer Data'][0]

# COMMAND ----------

install_marketplace_tables(listing=simulated_retail_listing, catalog_suffix="retail")

# COMMAND ----------

# Uncomment to delete the retail tables
# delete_marketplace_tables(listing_id=simulated_retail_listing.id, catalog_suffix="retail")

# COMMAND ----------

australia_sales_listing =  [marketplace_list for marketplace_list in w.consumer_listings.list() if marketplace_list.summary.name=='Simulated Australia Sales and Opportunities Data'][0]

# COMMAND ----------

install_marketplace_tables(listing=australia_sales_listing, catalog_suffix="australia_sales")

# COMMAND ----------

# Uncomment to delete the retail tables
# delete_marketplace_tables(listing_id=australia_sales_listing.id, catalog_suffix="australia_sales")

# COMMAND ----------

# MAGIC %md
# MAGIC ![Create Genie Room step 1](./create_genie_room1.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ![Create Genie Room step 1](./create_genie_room2.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ![Create Genie Room step 1](./create_genie_room3.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ![Create Genie Room step 1](./create_genie_room4.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ![Create Genie Room step 1](./create_genie_room5.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ![Create Genie Room step 1](./create_genie_room5.png)

# COMMAND ----------


