// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://triggerdaq-docs.vercel.app',
  integrations: [
    starlight({
      title: 'TriggerDAQ',
      description:
        'Hybrid C++20 / Python Two-Stage Real-Time Detector Trigger & Data Acquisition Engine modeled on CERN ATLAS/CMS L1 + HLT.',
      logo: {
        dark: './src/assets/logo-dark.svg',
        light: './src/assets/logo-light.svg',
        replacesTitle: false,
      },
      favicon: '/favicon.svg',
      social: {
        github: 'https://github.com/OmarAhmedTHE25th/TriggerDAQ',
      },
      customCss: ['./src/styles/custom.css'],
      head: [
        {
          tag: 'link',
          attrs: {
            rel: 'preconnect',
            href: 'https://fonts.googleapis.com',
          },
        },
        {
          tag: 'link',
          attrs: {
            rel: 'preconnect',
            href: 'https://fonts.gstatic.com',
            crossorigin: '',
          },
        },
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap',
          },
        },
        {
          tag: 'script',
          content: `
            (() => {
              const root = document.documentElement;
              const L_KEY = 'tdaq-left-collapsed';
              const R_KEY = 'tdaq-right-collapsed';
              try {
                if (localStorage.getItem(L_KEY) === 'true') root.dataset.leftCollapsed = 'true';
                if (localStorage.getItem(R_KEY) === 'true') root.dataset.rightCollapsed = 'true';
              } catch {}

              const panelLeftIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>';
              const panelRightIcon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="15" y1="3" x2="15" y2="21"/></svg>';

              function togglePane(side) {
                const attr = side === 'left' ? 'leftCollapsed' : 'rightCollapsed';
                const key = side === 'left' ? L_KEY : R_KEY;
                const next = root.dataset[attr] !== 'true';
                root.dataset[attr] = String(next);
                try { localStorage.setItem(key, String(next)); } catch {}
                updateButtons();
              }

              function updateButtons() {
                const lBtn = document.getElementById('tdaq-toggle-left');
                const rBtn = document.getElementById('tdaq-toggle-right');
                if (lBtn) {
                  const collapsed = root.dataset.leftCollapsed === 'true';
                  lBtn.setAttribute('aria-expanded', String(!collapsed));
                  lBtn.title = (collapsed ? 'Expand' : 'Collapse') + ' navigation sidebar ([)';
                }
                if (rBtn) {
                  const collapsed = root.dataset.rightCollapsed === 'true';
                  rBtn.setAttribute('aria-expanded', String(!collapsed));
                  rBtn.title = (collapsed ? 'Expand' : 'Collapse') + ' table of contents (])';
                }
              }

              document.addEventListener('DOMContentLoaded', () => {
                if (root.hasAttribute('data-has-sidebar') && !document.getElementById('tdaq-toggle-left')) {
                  const btn = document.createElement('button');
                  btn.id = 'tdaq-toggle-left';
                  btn.type = 'button';
                  btn.className = 'tdaq-pane-toggle left';
                  btn.innerHTML = panelLeftIcon;
                  btn.addEventListener('click', () => togglePane('left'));
                  document.body.appendChild(btn);
                }
                if (root.hasAttribute('data-has-toc') && !document.getElementById('tdaq-toggle-right')) {
                  const btn = document.createElement('button');
                  btn.id = 'tdaq-toggle-right';
                  btn.type = 'button';
                  btn.className = 'tdaq-pane-toggle right';
                  btn.innerHTML = panelRightIcon;
                  btn.addEventListener('click', () => togglePane('right'));
                  document.body.appendChild(btn);
                }
                updateButtons();
              });

              document.addEventListener('keydown', (e) => {
                const tag = document.activeElement && document.activeElement.tagName;
                if (tag === 'INPUT' || tag === 'TEXTAREA' || (document.activeElement && document.activeElement.isContentEditable)) return;
                if (e.metaKey || e.ctrlKey || e.altKey) return;
                if (e.key === '[' && root.hasAttribute('data-has-sidebar')) togglePane('left');
                if (e.key === ']' && root.hasAttribute('data-has-toc')) togglePane('right');
              });
            })();
          `,
        },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Installation & Build', slug: 'getting-started/installation' },
            { label: 'Quickstart & CLI', slug: 'getting-started/quickstart' },
          ],
        },
        {
          label: 'Architecture',
          items: [
            { label: 'System Overview', slug: 'architecture/overview' },
            { label: 'Memory & Lock-Free Concurrency', slug: 'architecture/memory-and-concurrency' },
            { label: 'CERN Domain Modeling', slug: 'architecture/cern-domain' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Trigger Menus & PI Prescaling', slug: 'guides/trigger-menus' },
            { label: 'Binary .daq Storage & Zero-Suppression', slug: 'guides/storage-and-daq' },
          ],
        },
        {
          label: 'Reference',
          items: [
            { label: 'Python API & CLI Flags', slug: 'reference/python-api' },
            { label: 'Benchmarks & Sanitizer CI', slug: 'reference/benchmarks-and-sanitizers' },
          ],
        },
      ],
    }),
  ],
});
