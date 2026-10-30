from triage.fetchers import ashby, greenhouse, lever, recruitee, teamtailor, workable
from triage.fetchers.base import Fetcher

FETCHERS: dict[str, Fetcher] = {
    "greenhouse": greenhouse.fetch,
    "lever": lever.fetch,
    "ashby": ashby.fetch,
    "recruitee": recruitee.fetch,
    "workable": workable.fetch,
    "teamtailor": teamtailor.fetch,
}
