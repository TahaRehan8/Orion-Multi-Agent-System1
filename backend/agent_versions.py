"""
Agent Version History for Orion Multi-Agent RAG System
Tracks version history and supports rollback for custom agents
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class AgentVersion:
    """A versioned snapshot of an agent configuration"""
    version: int
    agent_id: str
    prompt_template: str
    tools: List[str]
    description: str
    created_at: str
    created_by: str
    change_note: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


# Storage
VERSIONS_PATH = Path(__file__).parent.parent / "outputs" / "agent_versions.json"
_versions: Dict[str, List[AgentVersion]] = {}


def _load_versions():
    """Load versions from disk"""
    global _versions
    if VERSIONS_PATH.exists():
        try:
            with open(VERSIONS_PATH, "r") as f:
                data = json.load(f)
                for agent_id, versions in data.items():
                    _versions[agent_id] = [AgentVersion(**v) for v in versions]
        except Exception as e:
            print(f"[VERSIONS] Error loading versions: {e}")


def _save_versions():
    """Save versions to disk"""
    try:
        VERSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            agent_id: [v.to_dict() for v in versions]
            for agent_id, versions in _versions.items()
        }
        with open(VERSIONS_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[VERSIONS] Error saving versions: {e}")


_load_versions()


def save_version(
    agent_id: str,
    prompt_template: str,
    tools: List[str],
    description: str,
    created_by: str,
    change_note: str = None
) -> AgentVersion:
    """Save a new version of an agent"""
    if agent_id not in _versions:
        _versions[agent_id] = []
    
    version_num = len(_versions[agent_id]) + 1
    
    version = AgentVersion(
        version=version_num,
        agent_id=agent_id,
        prompt_template=prompt_template,
        tools=tools,
        description=description,
        created_at=datetime.now().isoformat(),
        created_by=created_by,
        change_note=change_note
    )
    
    _versions[agent_id].append(version)
    _save_versions()
    
    print(f"[VERSIONS] Saved version {version_num} for agent {agent_id}")
    return version


def list_versions(agent_id: str) -> List[AgentVersion]:
    """List all versions of an agent"""
    return _versions.get(agent_id, [])


def get_version(agent_id: str, version_num: int) -> Optional[AgentVersion]:
    """Get a specific version of an agent"""
    versions = _versions.get(agent_id, [])
    for v in versions:
        if v.version == version_num:
            return v
    return None


def get_latest_version(agent_id: str) -> Optional[AgentVersion]:
    """Get the latest version of an agent"""
    versions = _versions.get(agent_id, [])
    return versions[-1] if versions else None


def restore_version(agent_id: str, version_num: int, restored_by: str) -> bool:
    """Restore an agent to a previous version"""
    from backend.registry import get_agent, update_agent_prompt
    
    version = get_version(agent_id, version_num)
    if not version:
        return False
    
    agent = get_agent(agent_id)
    if not agent or not agent.is_custom:
        return False
    
    # Save current state as a new version before restoring
    save_version(
        agent_id=agent_id,
        prompt_template=agent.prompt_template or "",
        tools=agent.tools,
        description=agent.description,
        created_by=restored_by,
        change_note=f"Auto-saved before restoring to version {version_num}"
    )
    
    # Apply the old version
    success = update_agent_prompt(agent_id, version.prompt_template)
    
    if success:
        save_version(
            agent_id=agent_id,
            prompt_template=version.prompt_template,
            tools=version.tools,
            description=version.description,
            created_by=restored_by,
            change_note=f"Restored from version {version_num}"
        )
        print(f"[VERSIONS] Agent {agent_id} restored to version {version_num}")
    
    return success


def get_default_prompt(agent_id: str) -> Optional[str]:
    """Get the original/default prompt (version 1)"""
    version = get_version(agent_id, 1)
    return version.prompt_template if version else None


def restore_to_default(agent_id: str, restored_by: str) -> bool:
    """Restore an agent to its original configuration"""
    return restore_version(agent_id, 1, restored_by)


def get_version_diff(agent_id: str, version_a: int, version_b: int) -> Dict:
    """Compare two versions of an agent"""
    v_a = get_version(agent_id, version_a)
    v_b = get_version(agent_id, version_b)
    
    if not v_a or not v_b:
        return {"error": "One or both versions not found"}
    
    return {
        "version_a": version_a,
        "version_b": version_b,
        "prompt_changed": v_a.prompt_template != v_b.prompt_template,
        "tools_changed": set(v_a.tools) != set(v_b.tools),
        "tools_added": list(set(v_b.tools) - set(v_a.tools)),
        "tools_removed": list(set(v_a.tools) - set(v_b.tools)),
        "description_changed": v_a.description != v_b.description
    }


if __name__ == "__main__":
    # Test the version history module
    print("Testing version history...")
    
    # Simulate saving versions
    v1 = save_version(
        agent_id="test_agent",
        prompt_template="You are a helpful assistant.",
        tools=["search", "export"],
        description="Test agent",
        created_by="test_user",
        change_note="Initial version"
    )
    print(f"Created version {v1.version}")
    
    v2 = save_version(
        agent_id="test_agent",
        prompt_template="You are a specialized financial assistant.",
        tools=["search", "export", "forecast"],
        description="Updated test agent",
        created_by="test_user",
        change_note="Added forecasting"
    )
    print(f"Created version {v2.version}")
    
    print(f"\nAll versions: {[v.version for v in list_versions('test_agent')]}")
