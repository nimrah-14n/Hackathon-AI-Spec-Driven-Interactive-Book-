import React from 'react';
import clsx from 'clsx';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import BlogListPaginator from '@theme/BlogListPaginator';
import BlogPostItems from '@theme/BlogPostItems';
import BlogPostItem from '@site/src/components/Blog/BlogPostItem';

import styles from './blog-list.module.css';

function BlogListPage(props) {
  const {metadata, items} = props;
  const {siteConfig} = useDocusaurusContext();
  const {blogTitle, blogDescription} = metadata;

  return (
    <Layout
      title={blogTitle}
      description={blogDescription}>
      <div className={clsx('container margin-vert--lg', styles.blogListPage)}>
        <div className="row">
          <main className="col col--8 col--offset-2">
            <header className="hero text--center">
              <h1 className="hero__title">{blogTitle}</h1>
              <p className="hero__subtitle">{blogDescription}</p>
            </header>
            <div className="margin-vert--lg">
              <BlogPostItems>
                {items.map(({content: BlogPostContent}) => (
                  <BlogPostItem
                    key={BlogPostContent.metadata.permalink}
                    frontMatter={BlogPostContent.frontMatter}
                    metadata={BlogPostContent.metadata}
                    truncated={BlogPostContent.metadata.truncated}>
                    <BlogPostContent />
                  </BlogPostItem>
                ))}
              </BlogPostItems>
            </div>
            <BlogListPaginator metadata={metadata} />
          </main>
        </div>
      </div>
    </Layout>
  );
}

export default BlogListPage;