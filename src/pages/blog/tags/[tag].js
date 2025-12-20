import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import BlogPostItems from '@theme/BlogPostItems';
import BlogPostItem from '@site/src/components/Blog/BlogPostItem';

import styles from './blog-tag.module.css';

function BlogTagPage(props) {
  const {metadata, items} = props;
  const {tag, blogPosts} = metadata;

  const {siteConfig} = useDocusaurusContext();
  const blogTitle = siteConfig.title + ' Blog';

  return (
    <Layout
      title={`Blog | Tagged "${tag.label}"`}
      description={`Blog posts tagged with "${tag.label}"`}>
      <div className={clsx('container margin-vert--lg', styles.blogTagPage)}>
        <div className="row">
          <main className="col col--8 col--offset-2">
            <header className="text--center margin-bottom--xl">
              <h1>
                #{tag.label} ({blogPosts.length} post{blogPosts.length !== 1 ? 's' : ''})
              </h1>
              <Link to="/blog/tags">View all tags</Link>
            </header>
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
          </main>
        </div>
      </div>
    </Layout>
  );
}

export default BlogTagPage;