"""
Authentication module for getting Graph and Substrate tokens.

This module provides authentication for Microsoft services using MSAL and Azure CLI.
"""

import logging
import sys
import time
import msal
import jwt
import ctypes
from abc import ABC, abstractmethod

from azure.identity import AzureCliCredential

# Microsoft service configuration
_client_id = 'd3590ed6-52b3-4102-aeff-aad2292ab01c' # office client id
_tenant_id = '72f988bf-86f1-41af-91ab-2d7cd011db47' # msit tenant
_authority = f'https://login.microsoftonline.com/{_tenant_id}'
_scope = ['https://outlook.office.com/.default']
_graph_scope = ['https://graph.microsoft.com/.default']

# TODO: use our own client id for graph api
_mcp_client_id = '9ce97a32-d9ab-4ab2-aadc-f49b39b94e11'
# Sample LLM API client id
_sample_client_id = '68df66a4-cad9-4bfd-872b-c6ddde00d6b2'  # Sample LLM API client id
_substrate_llm_scopes = ["https://substrate.office.com/llmapi/LLMAPI.dev"]


class AuthProvider(ABC):
    """
    Abstract base class for authentication token generation
    """

    @abstractmethod
    def get_azure_openai_token(self) -> str:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_substrate_token(self) -> str:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_substrate_llm_token(self) -> str:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def get_graph_token(self) -> str:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def refresh_azure_openai_token(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    @abstractmethod
    def refresh_substrate_token(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def refresh_substrate_llm_token(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @abstractmethod
    def refresh_graph_token(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")

    def ensure_all_tokens(self) -> None:
        """
        Refresh all tokens
        """
        self.get_azure_openai_token()
        self.get_graph_token()
        self.get_substrate_token()
        self.get_substrate_llm_token()


class DefaultAuthProvider(AuthProvider):
    """
    Default implementation of the AuthProvider class. This is not great for server scenarios as it caches tokens.
    """

    def __init__(self) -> None:
        self._app = msal.PublicClientApplication(
            _client_id,
            authority=_authority,
            enable_broker_on_windows=True,
        )

        self._mcp_app = msal.PublicClientApplication(
            _mcp_client_id,
            authority=_authority,
            enable_broker_on_windows=True,
        )

        self._sample_llm_app = msal.PublicClientApplication(
            _sample_client_id,
            authority=_authority,
            enable_broker_on_windows=True,
        )

        self._tokens = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _set_token(self, name: str, token: str) -> None:
        self._tokens[name] = token

    def _get_token(self, name: str) -> str | None:
        return self._tokens.get(name, None)

    def _is_fresh(self, token: str | None) -> bool:
        if token is None:
            self.logger.info("Token is None")
            return False

        if token is not None:
            decoded_token = jwt.decode(token, options={"verify_signature": False})
            if decoded_token['exp'] < time.time():
                self.logger.info(f"Token expired at {decoded_token['exp']}, current time is {time.time()}")
                return False
            
        return True

    def get_azure_openai_token(self) -> str:
        self.refresh_azure_openai_token()
        return self._get_token("azure")
    def get_azure_openai_token(self) -> str:
        self.refresh_azure_openai_token()
        return self._get_token("azure")

    def get_substrate_token(self) -> str:
        self.refresh_substrate_token()
        return self._get_token("substrate")
    def get_substrate_token(self) -> str:
        self.refresh_substrate_token()
        return self._get_token("substrate")

    def get_substrate_llm_token(self) -> str:
        self.refresh_substrate_llm_token()
        return self._get_token("substrate_llm")
    def get_substrate_llm_token(self) -> str:
        self.refresh_substrate_llm_token()
        return self._get_token("substrate_llm")

    def get_graph_token(self) -> str:
        self.refresh_graph_token()
        return self._get_token("graph")   
    def get_graph_token(self) -> str:
        self.refresh_graph_token()
        return self._get_token("graph")   

    def refresh_substrate_token(self) -> None:
        current_token = self._get_token("substrate")
        if self._is_fresh(current_token):
            return
        self.logger.info("Refreshing substrate token")
        # Acquire a token interactively
        result = self._app.acquire_token_interactive(scopes=_scope, parent_window_handle=self._get_console_window())
        if 'access_token' in result:
            self._set_token("substrate", result['access_token'])
        else:
            raise Exception("Failed to acquire token")
    def refresh_substrate_token(self) -> None:
        current_token = self._get_token("substrate")
        if self._is_fresh(current_token):
            return
                
        self.logger.info("Refreshing substrate token")
        # Acquire a token interactively
        result = self._app.acquire_token_interactive(scopes=_scope, parent_window_handle=self._get_console_window())
        if 'access_token' in result:
            self._set_token("substrate", result['access_token'])
        else:
            raise Exception("Failed to acquire token")

    def refresh_substrate_llm_token(self) -> None:
        current_token = self._get_token("substrate_llm")
        if self._is_fresh(current_token):
            return
        self.logger.info("Refreshing substrate llm token")
        # Acquire a token interactively
        result = self._sample_llm_app.acquire_token_interactive(scopes=_substrate_llm_scopes, parent_window_handle=self._get_console_window())
        if 'access_token' in result:
            self._set_token("substrate_llm", result['access_token'])
        else:
            raise Exception("Failed to acquire token")
    def refresh_substrate_llm_token(self) -> None:
        current_token = self._get_token("substrate_llm")
        if self._is_fresh(current_token):
            return
                
        self.logger.info("Refreshing substrate llm token")
        # Acquire a token interactively
        result = self._sample_llm_app.acquire_token_interactive(scopes=_substrate_llm_scopes, parent_window_handle=self._get_console_window())
        if 'access_token' in result:
            self._set_token("substrate_llm", result['access_token'])
        else:
            raise Exception("Failed to acquire token")

    def refresh_azure_openai_token(self) -> None:
        current_token = self._get_token("azure")
        if self._is_fresh(current_token):
            return
        self.logger.info("Refreshing oai token")
        credential = AzureCliCredential()
        current_token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        self._set_token("azure", current_token)
    def refresh_azure_openai_token(self) -> None:
        current_token = self._get_token("azure")
        if self._is_fresh(current_token):
            return
        
        self.logger.info("Refreshing oai token")
        credential = AzureCliCredential()
        current_token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        self._set_token("azure", current_token)

    def refresh_graph_token(self) -> None:
        current_token = self._get_token("graph")
        if self._is_fresh(current_token):
            return
        self.logger.info("Refreshing graph token")
        # Acquire a token interactively
        result = self._mcp_app.acquire_token_interactive(scopes=_graph_scope, parent_window_handle=self._get_console_window())
        if 'access_token' in result:
            self._set_token('graph', result['access_token'])
        else:
            raise Exception("Failed to acquire token")
    def refresh_graph_token(self) -> None:
        current_token = self._get_token("graph")
        if self._is_fresh(current_token):
            return

        self.logger.info("Refreshing graph token")
        # Acquire a token interactively
        result = self._mcp_app.acquire_token_interactive(scopes=_graph_scope, parent_window_handle=self._get_console_window())

        if 'access_token' in result:
            self._set_token('graph', result['access_token'])
        else:
            raise Exception("Failed to acquire token")

    def _get_console_window(self) -> None:
        """
        Gets the console window handle on Windows. Returns None on other platforms.
        This is used to center the interactive login window.
        """
        if sys.platform != "win32":
            self.logger.warning("Console window handle is only available on Windows.\nPlease consider using InputAuth if auth failed.")
            return None
    
        try:
            return ctypes.windll.kernel32.GetConsoleWindow()
        except (AttributeError, OSError) as e:
            self.logger.warning(f"Could not get console window handle: {e}")
            return None

def get_shard_id(token):
    """Get shard ID from token."""
    decoded_token = jwt.decode(token, options={"verify_signature": False})
    return "OID:" + decoded_token['oid'] + "@" + decoded_token['tid']


def get_tenant_id(token):
    """Get tenant ID from token."""
    decoded_token = jwt.decode(token, options={"verify_signature": False})
    return decoded_token['tid']
 

def get_user_id(token):
    """Get user ID from token."""
    decoded_token = jwt.decode(token, options={"verify_signature": False})
    return decoded_token['oid']


# Simple wrapper class for easy usage
class SimpleAuth:
    """Simple authentication class for getting Microsoft service tokens."""
    
    def __init__(self):
        self.auth_provider = DefaultAuthProvider()
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def get_graph_token(self) -> str:
        """Get a token for Microsoft Graph API."""
        return self.auth_provider.get_graph_token()
    
    def get_substrate_token(self) -> str:
        """Get a token for Substrate service."""
        return self.auth_provider.get_substrate_token()
    
    def get_azure_openai_token(self) -> str:
        """Get a token for Azure OpenAI service."""
        return self.auth_provider.get_azure_openai_token()
    
    def get_substrate_llm_token(self) -> str:
        """Get a token for Substrate LLM service."""
        return self.auth_provider.get_substrate_llm_token()
    
    def clear_cache(self):
        """Clear all cached tokens."""
        self.auth_provider._tokens.clear()
        self.logger.info("Cleared token cache")
        

class InputAuth(AuthProvider):
    """
    Input-based authentication provider for testing purposes.
    This class prompts the user for tokens instead of acquiring them automatically.
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        self.substrate_token = None
        self.graph_token = None
        self.azure_openai_token = None
        self.substrate_llm_token = None

    def get_azure_openai_token(self) -> str:
        # There is non good way to get Azure OpenAI token without user input
        raise NotImplementedError("Azure OpenAI token not found a good way to get it, please try to use SimpleAuth or DefaultAuthProvider")

    def get_substrate_token(self) -> str:
        if self.substrate_token and self._valid(self.substrate_token):
            return self.substrate_token
        self.logger.info("Substrate token is not set or invalid, prompting for input.")
        self.substrate_token = input("Enter Substrate token: ")
        return self.substrate_token

    def get_substrate_llm_token(self) -> str:
        if self.substrate_llm_token and self._valid(self.substrate_llm_token):
            return self.substrate_llm_token
        self.logger.info("Substrate LLM token is not set or invalid, prompting for input.")
        self.substrate_llm_token = input("Enter Substrate LLM token: ")
        return self.substrate_llm_token

    def get_graph_token(self) -> str:
        if self.graph_token and self._valid(self.graph_token):
            return self.graph_token
            
        self.graph_token = input("Enter Microsoft Graph token: ")
        return self.graph_token

    def refresh_azure_openai_token(self) -> None:
        pass

    def refresh_substrate_token(self) -> None:
        pass

    def refresh_substrate_llm_token(self) -> None:
        pass

    def refresh_graph_token(self) -> None:
        pass

    def _valid(self, token: str) -> bool:
        if token is None:
            self.logger.info("Token is None")
            return False

        if token is not None:
            decoded_token = jwt.decode(token, options={"verify_signature": False})
            if decoded_token['exp'] < time.time():
                self.logger.info(f"Token expired at {decoded_token['exp']}, current time is {time.time()}")
                return False
            
        return True

auth = SimpleAuth()

# Convenience functions for easy import
def get_graph_token() -> str:
    """Get a Microsoft Graph token."""
    return auth.get_graph_token()


def get_substrate_token() -> str:
    """Get a Substrate service token."""
    return auth.get_substrate_token()


def get_substrate_llm_token() -> str:
    """Get a Substrate LLM service token."""
    return auth.get_substrate_llm_token()


def get_azure_openai_token() -> str:
    """Get an Azure OpenAI token."""
    return auth.get_azure_openai_token()


if __name__ == "__main__":
    # Example usage
    try:
        graph_token = get_graph_token()
        
        # query the user profile
        import requests
        headers = {
            'Authorization': f'Bearer {graph_token}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get('https://graph.microsoft.com/v1.0/me', headers=headers)
        if response.status_code == 200:
            user_profile = response.json()
            print(f"hello {user_profile.get('displayName', 'user')}, test graph token succeeded")
        else:
            print("Failed to fetch user profile:", response.status_code, response.text)
    except Exception as e:
        print("Error:", str(e))
        logging.exception("An error occurred while fetching the Graph token or user profile.")