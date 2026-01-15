'use client';

import { cn } from '@/lib/utils';
import type { TabId, TabConfig } from '@/types';

interface TabsProps {
  tabs: TabConfig[];
  activeTab: TabId;
  onChange: (tab: TabId) => void;
}

export function Tabs({ tabs, activeTab, onChange }: TabsProps) {
  return (
    <div className="flex gap-2 p-1 bg-surface rounded-xl">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={cn(
            'tab-button flex-1 sm:flex-none',
            activeTab === tab.id && 'tab-button-active'
          )}
        >
          <span className="block sm:hidden">{tab.label.slice(0, 3)}</span>
          <span className="hidden sm:block">{tab.label}</span>
        </button>
      ))}
    </div>
  );
}
