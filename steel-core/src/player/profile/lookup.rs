//! Online player name-to-identity lookup.

#[cfg(feature = "profile-lookup")]
use reqwest::{StatusCode, Url};
#[cfg(feature = "profile-lookup")]
use serde::Deserialize;
use thiserror::Error;
#[cfg(feature = "profile-lookup")]
use tokio::time::{Duration, sleep};
#[cfg(feature = "profile-lookup")]
use uuid::Uuid;

#[cfg(feature = "profile-lookup")]
use super::known_players::KnownPlayer;

#[cfg(feature = "profile-lookup")]
const DEFAULT_PROFILE_SERVER: &str =
    "https://api.minecraftservices.com/minecraft/profile/lookup/name";
#[cfg(feature = "profile-lookup")]
const MAX_PROFILE_LOOKUP_ATTEMPTS: usize = 3;
#[cfg(feature = "profile-lookup")]
const PROFILE_LOOKUP_RETRY_DELAY: Duration = Duration::from_millis(750);
/// Bounds every attempt so suspended administrative commands always release their ordering barrier.
#[cfg(feature = "profile-lookup")]
const PROFILE_LOOKUP_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Failure while resolving a player identity through the configured profile service.
#[derive(Debug, Error)]
pub enum ProfileLookupError {
    /// No profile exists for the requested name.
    #[error("Unknown player {0}")]
    UnknownPlayer(String),
    /// The configured profile lookup endpoint is invalid.
    #[error("Invalid profile server URL configured: {0}")]
    InvalidProfileServer(String),
    /// The request failed before a response was received.
    #[cfg(feature = "profile-lookup")]
    #[error("Profile lookup failed for {name}: {source}")]
    Request {
        /// Requested player name.
        name: String,
        /// Transport error.
        source: reqwest::Error,
    },
    /// The service returned an unexpected status.
    #[cfg(feature = "profile-lookup")]
    #[error("Profile lookup service returned status {status} for {name}")]
    ServiceResponse {
        /// Requested player name.
        name: String,
        /// HTTP response status.
        status: StatusCode,
    },
    /// This build cannot query the profile service.
    ///
    /// Only produced when the `profile-lookup` feature is off. Kept a distinct
    /// variant rather than reusing `UnknownPlayer` so callers can tell "no such
    /// player" from "this build cannot ask".
    #[cfg(not(feature = "profile-lookup"))]
    #[error("Online profile lookup is unavailable in this build (requested {0})")]
    LookupUnavailable(String),
    /// The service returned malformed identity data.
    #[error("Invalid profile lookup response for {name}: {reason}")]
    InvalidResponse {
        /// Requested player name.
        name: String,
        /// Response validation failure.
        reason: String,
    },
}

#[cfg(feature = "profile-lookup")]
#[derive(Deserialize)]
struct ProfileLookupResponse {
    id: String,
    name: String,
}

/// Resolves one online-mode profile through the configured service.
///
/// The caller handles local caches, offline mode, and name validation first.
#[cfg(feature = "profile-lookup")]
pub async fn lookup_online_profile(
    client: &reqwest::Client,
    profile_server: Option<&str>,
    name: &str,
) -> Result<KnownPlayer, ProfileLookupError> {
    let lookup_name = name.to_ascii_lowercase();
    let url = profile_lookup_url(profile_server, &lookup_name)?;
    for attempt in 1..=MAX_PROFILE_LOOKUP_ATTEMPTS {
        let result =
            lookup_online_profile_once(client, url.as_str(), name, PROFILE_LOOKUP_REQUEST_TIMEOUT)
                .await;
        match result {
            Ok(profile) => return Ok(profile),
            Err(error @ ProfileLookupError::UnknownPlayer(_)) => return Err(error),
            Err(error) if attempt == MAX_PROFILE_LOOKUP_ATTEMPTS => return Err(error),
            Err(_) => sleep(PROFILE_LOOKUP_RETRY_DELAY).await,
        }
    }
    unreachable!("the profile lookup attempt range is non-empty")
}

#[cfg(feature = "profile-lookup")]
async fn lookup_online_profile_once(
    client: &reqwest::Client,
    url: &str,
    name: &str,
    request_timeout: Duration,
) -> Result<KnownPlayer, ProfileLookupError> {
    let response = client
        .get(url)
        .timeout(request_timeout)
        .send()
        .await
        .map_err(|source| ProfileLookupError::Request {
            name: name.to_owned(),
            source,
        })?;

    match response.status() {
        StatusCode::OK => parse_profile_response(response, name).await,
        StatusCode::NO_CONTENT | StatusCode::NOT_FOUND => {
            Err(ProfileLookupError::UnknownPlayer(name.to_owned()))
        }
        status => Err(ProfileLookupError::ServiceResponse {
            name: name.to_owned(),
            status,
        }),
    }
}

#[cfg(feature = "profile-lookup")]
fn profile_lookup_url(
    profile_server: Option<&str>,
    normalized_name: &str,
) -> Result<Url, ProfileLookupError> {
    let server = profile_server.unwrap_or(DEFAULT_PROFILE_SERVER);
    let endpoint = format!("{}/{normalized_name}", server.trim_end_matches('/'));
    Url::parse(&endpoint).map_err(|_| ProfileLookupError::InvalidProfileServer(endpoint))
}

#[cfg(feature = "profile-lookup")]
async fn parse_profile_response(
    response: reqwest::Response,
    requested_name: &str,
) -> Result<KnownPlayer, ProfileLookupError> {
    let profile = response
        .json::<ProfileLookupResponse>()
        .await
        .map_err(|source| ProfileLookupError::InvalidResponse {
            name: requested_name.to_owned(),
            reason: source.to_string(),
        })?;
    let uuid =
        Uuid::parse_str(&profile.id).map_err(|source| ProfileLookupError::InvalidResponse {
            name: requested_name.to_owned(),
            reason: source.to_string(),
        })?;
    Ok(KnownPlayer::new(uuid, profile.name))
}

#[cfg(test)]
mod tests {
    use super::profile_lookup_url;

    #[test]
    fn profile_lookup_url_uses_mojangs_default_endpoint() {
        let url = profile_lookup_url(None, "steve");
        let Ok(url) = url else {
            panic!("default profile lookup URL should build");
        };
        assert_eq!(
            url.as_str(),
            "https://api.minecraftservices.com/minecraft/profile/lookup/name/steve"
        );
    }

    #[test]
    fn profile_lookup_url_uses_the_configured_endpoint() {
        let url = profile_lookup_url(Some("https://profiles.example.com/lookup/"), "steve");
        let Ok(url) = url else {
            panic!("configured profile lookup URL should build");
        };
        assert_eq!(url.as_str(), "https://profiles.example.com/lookup/steve");
    }
}
