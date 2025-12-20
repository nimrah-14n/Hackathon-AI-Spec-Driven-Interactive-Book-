import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './blog-tags.module.css';

function BlogTagsListPage(props) {
  const {metadata} = props;
  const {tags} = metadata;

  const {siteConfig} = useDocusaurusContext();
  const blogTitle = siteConfig.title + ' Blog';

  return (
    <Layout
      title={`Blog Tags | ${blogTitle}`}
      description="Tags used in blog posts">
      <div className={clsx('container margin-vert--lg', styles.blogTagsListPage)}>
        <div className="row">
          <main className="col col--8 col--offset-2">
            <header className="text--center">
              <h1>Blog Tags</h1>
            </header>
            <div className="tag-cloud">
              {tags.map((tag) => (
                <Link
                  key={tag.permalink}
                  className={clsx('badge badge--lg margin-horiz--md blog-tag', styles.tag)}
                  to={tag.permalink}>
                  #{tag.label} ({tag.count})
                </Link>
              ))}
            </div>
          </main>
        </div>
      </div>
    </Layout>
  );
}

export default BlogTagsListPage;