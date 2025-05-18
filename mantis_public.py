from Mantis_SDK.mantis_sdk.client import MantisClient, SpacePrivacy, DataType, ReducerModels 
# from Mantis_SDK.mantis_sdk.render_args import RenderArgs 
import pandas as pd

from agent.ranker import process_amazon_results
from openai import OpenAI
from agent.retriever import retrieve_products_agent

cookie = ""
url = "https://mantisdev.csail.mit.edu"

async def main():
    # mantis = MantisClient(f"{url}/api/proxy/", cookie)
    mantis = MantisClient("/api/proxy/", cookie)
    client = OpenAI(api_key="")

    query = "find me a coffee maker with 1000+ amazon reviews and made in Italy"
    query = "Find me a coffee maker under $100 with good reviews that's available for Amazon Prime shipping"
    query = "Electrolyte supplement with no sugar and no artificial sweeteners, but still flavored with stevia or monk fruit. Preferably in powder form, but capsules are ok too. I want to avoid any artificial colors or flavors. No caffeine, bcaas, or additional workout ingredients, just electrolytes. I want to be able to find it on Amazon."

    print(f"Retrieving products for query: {query}")
    products, func = retrieve_products_agent(query, client, debug=True)
    print(f"Finished retrieving for: {query}")
    print(f"Function call: {func}")
    print(f"Products: {products}")
    # Process results
    print("Processing results...")
    if "amazon" in func:
        processed = process_amazon_results(products)
    else:
        raise ValueError(f"Did not use Amazon, used: {func}")
    print("Finished processing results...")

    df = pd.DataFrame(processed)
    df_subset = df[["title", "price", "description", "link", "reviews", "rating", "review_content", "at_a_glance"]]
    data_types = {"title": DataType.Title,
                "price": DataType.Numeric,
                "description": DataType.Semantic,
                "link": DataType.Links,
                "reviews": DataType.Numeric,
                "rating": DataType.Numeric,
                "review_content": DataType.Semantic,
                "at_a_glance": DataType.Semantic,}


    # Make space
    print("Creating Mantis space...")
    new_space_id = mantis.create_space(f"Product Data for q: {query}", 
                                    data=df_subset, 
                                    data_types=data_types,
                                    reducer=ReducerModels.UMAP,
                                    privacy_level=SpacePrivacy.PRIVATE)["space_id"]
    raise Exception("NOTE: Mantis space creation succeeded!")
    # Open space
    # space = await mantis.open_space(new_space_id)

    # TODO: we found create_space always fails so this functionality was never used. Luckily, the space is created in Mantis, and we can access there.
    # Interact with space
    # await space.select_points(100) 
    # plot = await space.render_plot("Market Cap", "embed_y")

    # # Close when done
    # await space.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())