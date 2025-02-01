import os
import requests


def scrape_linked_profile(linkedin_profile_url: str, mock: bool = False):
    '''Scrape Information from LinkedIn Profiles. Manually scrape information
    from the linkedin profile'''

    if mock:

        linkedin_profile_url = os.environ["MOCK_LINKEDIN_URL"]
        response = requests.get(
            linkedin_profile_url,
            timeout=10
        )
    else:
        header_dict = {
            "Authorization": f'Bearer {os.getenv("PROXYCURL_API_KEY")}'}
        api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
        response = requests.get(api_endpoint,
                                params={"url": linkedin_profile_url},
                                headers=header_dict)

    data = response.json()

    # Clean redundant tokens
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


if __name__ == "__main__":
    print(
        scrape_linked_profile(
            # linkedin_profile_url="https://www.linkedin.com/in/subhankarbhattacharya/"
            linkedin_profile_url="https://www.linkedin.com/in/eden-marco/"
        )
    )
