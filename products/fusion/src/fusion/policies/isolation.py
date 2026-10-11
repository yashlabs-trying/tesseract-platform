from __future__ import annotations
from typing import Dict, List, Set, Tuple
from fusion.models.types import GPUInventory, TenantPolicy, IsolationMode, ConstraintResult, ConstraintCheck


class IsolationValidator:
    def check_tenant_compatibility(self, t1: TenantPolicy, t2: TenantPolicy) -> ConstraintCheck:
        if t1.isolation_requirements == IsolationMode.EXCLUSIVE or t2.isolation_requirements == IsolationMode.EXCLUSIVE:
            if t1.tenant_id != t2.tenant_id:
                return ConstraintCheck(
                    name="tenant_isolation",
                    result=ConstraintResult.FAIL,
                    details=f"Tenant {t1.tenant_id} requires exclusive, cannot co-locate with {t2.tenant_id}"
                )
        return ConstraintCheck(name="tenant_isolation", result=ConstraintResult.PASS)

    def find_safe_consolidation(
        self,
        tenants: Dict[str, TenantPolicy],
        gpu: GPUInventory,
    ) -> List[Set[str]]:
        groups: List[Set[str]] = []
        remaining = set(tenants.keys())
        while remaining:
            group = {remaining.pop()}
            changed = True
            while changed:
                changed = False
                for t in list(remaining):
                    if all(self.check_tenant_compatibility(tenants[t], tenants[g]).result == ConstraintResult.PASS
                           for g in group):
                        group.add(t)
                        remaining.remove(t)
                        changed = True
            groups.append(group)
        return groups
