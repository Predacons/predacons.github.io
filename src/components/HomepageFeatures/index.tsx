import React, { ReactNode } from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Predacons Library',
    Svg: require('@site/static/img/image.svg').default, // Replace with Predacons-specific SVG
    description: (
      <>
        A Python library based on transformers for transfer learning. Offers tools for data processing, model training, and text generation, simplifying the application of advanced ML techniques.
        <br />
        Key features: Data Loading, Text Cleaning, Model Training, Text Generation, Text & Chat Streaming, Embeddings. Compatible with Langchain.
      </>
    ),
  },
  {
    title: 'Predacons Agent',
    Svg: require('@site/static/img/image (3).svg').default, // Replace with Predacons Agent-specific SVG
    description: (
      <>
        An agentic AI library based on Predacons for data analysis using Python notebook agents, web scraping, vector databases, and decision-making.
      </>
    ),
  },
  {
    title: 'Predacons Server',  
    Svg: require('@site/static/img/image (5).svg').default, // Replace with Predacons Server-specific SVG
    description: (
      <>
        An OpenAI API-compatible server built on Predacons to host any Torch and Hugging Face LLM model.
        <br />
        Features: Model Hosting, API Key Authentication, Scalable Architecture, Easy Integration via REST API.
      </>
    ),
  },
  {
    title: 'Predacons CLI',
    Svg: require('@site/static/img/image (4).svg').default, // Replace with Predacons CLI-specific SVG
    description: (
      <>
        A command-line interface for interacting with the Predacons library.  Provides a way to load models, generate responses, and manage configurations from the terminal.
        <br />
        Features: Model Management, Interactive Chat (including with vector stores), Web Scraper, Configuration Management.
      </>
    ),
  },
  {
    title: 'Predacons GUI',
    Svg: require('@site/static/img/image (1).svg').default, // Replace with Predacons GUI-specific SVG
    description: (
      <>
        A Gradio-based frontend for Predacons. Provides a visual interface for loading, saving, training, and testing models (Hugging Face and custom).
      </>
    ),
  },
  
];

function Feature({ title, Svg, description }: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}