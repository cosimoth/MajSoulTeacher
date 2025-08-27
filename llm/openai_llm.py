from .auth import get_azure_openai_token

from langchain_openai.chat_models import AzureChatOpenAI


class AOAILLMClient:  
    def __init__(self, openai_account: str, deployment_name: str, model_name: str,  
                 api_version='2024-12-01-preview', temperature=0.1, max_tokens=15000, top_p=0.9):  
        # Construct the Azure OpenAI endpoint from account  
        self._endpoint = f'https://{openai_account}.openai.azure.com'  
        self._deployment_name = deployment_name  
        self._model_name = model_name  
        self._api_version = api_version  
        self._temperature = temperature  
        self._max_tokens = max_tokens  
        self._top_p = top_p  
  
        self.llm = AzureChatOpenAI(  
            azure_endpoint=f'{self._endpoint}/',  
            azure_ad_token_provider=get_azure_openai_token,  # Use AzureCliCredential for Azure AD token fetching  
            openai_api_version=self._api_version,  
            deployment_name=self._deployment_name,  
            model_name=self._model_name,  
            streaming=False,  
            request_timeout=300,  
            max_retries=3,  
            temperature=self._temperature,  
            max_tokens=self._max_tokens,  
            top_p=self._top_p,  
        )  
  
    def send_request(self, system: str, user: str = None) -> str:  
        """Send a prompt (system + user or system only) to the Azure OpenAI chat endpoint."""  
        messages = [{"role": "system", "content": system}]  
        if user:  
            messages.append({"role": "user", "content": user})  
        response = self.llm.invoke(messages)  
        return response.content

llm_client = AOAILLMClient(
    openai_account='calendarai-ds-sc',
    deployment_name='gpt-4o',
    model_name='gpt-4o',
)
llm_client.send_request("", "你是一个专业的日本麻将AI助手。")