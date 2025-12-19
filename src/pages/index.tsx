import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          AI-Native Robotics Learning Platform
        </Heading>
        <p className="hero__subtitle">Physical AI and Embodied Intelligence Learning Platform</p>
        <p className="hero__author">Author: Nimrah Hussain</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Start Learning
          </Link>
          <Link
            className="button button--secondary button--lg margin-left--sm"
            to="/docs/tutorialSidebar">
            View Modules
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Physical AI and Embodied Intelligence Learning Platform">
      <HomepageHeader />
      <main>
        <section className={styles.modulesSection}>
          <div className="container">
            <div className="row">
              <div className="col col--12">
                <Heading as="h2" className={clsx('margin-bottom--lg', styles.sectionTitle)}>
                  Modules Overview
                </Heading>
              </div>
            </div>
            <div className="row">
              <div className="col col--3 margin-bottom--lg">
                <Link to="/docs/module-1/chapter-1" className={clsx('card', styles.moduleCard, styles.clickableCard)}>
                  <div className="card__header">
                    <h3>🤖 ROS 2 Robotic Nervous System</h3>
                  </div>
                  <div className="card__body">
                    <p>Learn about ROS 2 fundamentals, nodes, topics, services, and actions for robotic applications.</p>
                  </div>
                </Link>
              </div>
              <div className="col col--3 margin-bottom--lg">
                <Link to="/docs/module2-digital-twin/digital-twin-physical-ai" className={clsx('card', styles.moduleCard, styles.clickableCard)}>
                  <div className="card__header">
                    <h3>🧩 Digital Twin Simulation</h3>
                  </div>
                  <div className="card__body">
                    <p>Explore simulation environments with Gazebo and Unity for robotics development and testing.</p>
                  </div>
                </Link>
              </div>
              <div className="col col--3 margin-bottom--lg">
                <Link to="/docs/module3-ai-brain/ai-brain-humanoid" className={clsx('card', styles.moduleCard, styles.clickableCard)}>
                  <div className="card__header">
                    <h3>🧠 AI-Robot Brain (NVIDIA Isaac)</h3>
                  </div>
                  <div className="card__body">
                    <p>Discover NVIDIA Isaac platform for AI-powered robotics with Isaac Sim and synthetic data.</p>
                  </div>
                </Link>
              </div>
              <div className="col col--3 margin-bottom--lg">
                <Link to="/docs/module4-vla/vla-paradigm" className={clsx('card', styles.moduleCard, styles.clickableCard)}>
                  <div className="card__header">
                    <h3>👁️🗣️钍 Vision-Language-Action Systems</h3>
                  </div>
                  <div className="card__body">
                    <p>Master Vision-Language-Action models for conversational robotics and embodied intelligence.</p>
                  </div>
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}