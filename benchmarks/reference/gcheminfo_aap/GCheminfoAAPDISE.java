/*
 * SPDX-FileCopyrightText: Copyright 2008-2014 Genentech Inc.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compatibility runner for the AAP DEFAULT8 and DISE implementations shipped
 * with Gobbi et al. (2015). The original source requires the proprietary
 * OpenEye OEChem Java library. This runner retains the published path hashing,
 * greedy atom assignment, sphere-exclusion seed selection, and final nearest-
 * seed assignment while accepting molecular graphs prepared by RDKit.
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public final class GCheminfoAAPDISE {
    private static final int MAX_ATOM_NUM = 108;
    private static final int MAX_ATOM_TYPE = 223; // first prime >= 2 * 108 + 1
    private static final int MAX_BOND_TYPE = 5;

    private static final class Edge {
        final int neighbor;
        final int bondIndex;

        Edge(int neighbor, int bondIndex) {
            this.neighbor = neighbor;
            this.bondIndex = bondIndex;
        }
    }

    private static final class Graph {
        final int inputIndex;
        final String priority;
        final int[] atomTypes;
        final int[] bondTypes;
        final List<Edge>[] adjacency;

        @SuppressWarnings("unchecked")
        Graph(int inputIndex, String priority, int[] atomTypes, int[][] bonds) {
            this.inputIndex = inputIndex;
            this.priority = priority;
            this.atomTypes = atomTypes;
            this.bondTypes = new int[bonds.length];
            this.adjacency = (List<Edge>[]) new List<?>[atomTypes.length];
            for (int atom = 0; atom < atomTypes.length; atom++) {
                adjacency[atom] = new ArrayList<Edge>();
            }
            for (int bond = 0; bond < bonds.length; bond++) {
                int left = bonds[bond][0];
                int right = bonds[bond][1];
                bondTypes[bond] = bonds[bond][2];
                adjacency[left].add(new Edge(right, bond));
                adjacency[right].add(new Edge(left, bond));
            }
        }
    }

    private static final class Descriptor {
        final Graph graph;
        final char[][] atomPaths;

        Descriptor(Graph graph, int maxBonds) {
            this.graph = graph;
            this.atomPaths = new char[graph.atomTypes.length][];
            PathGenerator generator = new PathGenerator(graph, maxBonds);
            for (int atom = 0; atom < graph.atomTypes.length; atom++) {
                atomPaths[atom] = generator.pathsFrom(atom);
            }
        }
    }

    /** Port of supplementary IAAPathGeneratorChar, including Java int overflow. */
    private static final class PathGenerator {
        final Graph graph;
        final int maxBonds;
        final boolean[] atomVisited;
        final boolean[] bondVisited;
        final ArrayList<Integer> paths = new ArrayList<Integer>();
        int depth;
        int currentPath;

        PathGenerator(Graph graph, int maxBonds) {
            this.graph = graph;
            this.maxBonds = maxBonds;
            this.atomVisited = new boolean[graph.atomTypes.length];
            this.bondVisited = new boolean[graph.bondTypes.length];
        }

        char[] pathsFrom(int startAtom) {
            paths.clear();
            atomVisited[startAtom] = true;
            depth = 1;
            for (Edge edge : graph.adjacency[startAtom]) {
                currentPath = graph.bondTypes[edge.bondIndex] * MAX_ATOM_TYPE;
                bondVisited[edge.bondIndex] = true;
                addPath(edge.neighbor);
                bondVisited[edge.bondIndex] = false;
            }
            atomVisited[startAtom] = false;

            char[] result = new char[paths.size()];
            for (int index = 0; index < paths.size(); index++) {
                result[index] = (char) paths.get(index).intValue();
            }
            Arrays.sort(result);
            return result;
        }

        void addPath(int atom) {
            atomVisited[atom] = true;
            int previousPath = currentPath;
            currentPath = (currentPath + graph.atomTypes[atom]) * MAX_BOND_TYPE;
            int thisPath = currentPath;

            if (depth < maxBonds) {
                depth++;
                for (Edge edge : graph.adjacency[atom]) {
                    if (bondVisited[edge.bondIndex]) {
                        continue;
                    }
                    currentPath = (thisPath + graph.bondTypes[edge.bondIndex]) * MAX_ATOM_TYPE;
                    if (atomVisited[edge.neighbor]) {
                        paths.add(Integer.valueOf(currentPath));
                    } else {
                        bondVisited[edge.bondIndex] = true;
                        addPath(edge.neighbor);
                        bondVisited[edge.bondIndex] = false;
                    }
                }
                depth--;
            }

            paths.add(Integer.valueOf(thisPath));
            currentPath = previousPath;
            atomVisited[atom] = false;
        }
    }

    private static final class AtomPair {
        final int left;
        final int right;
        final double similarity;

        AtomPair(int left, int right, double similarity) {
            this.left = left;
            this.right = right;
            this.similarity = similarity;
        }
    }

    private static final Comparator<AtomPair> ATOM_PAIR_ORDER = new Comparator<AtomPair>() {
        @Override
        public int compare(AtomPair left, AtomPair right) {
            int similarityOrder = Double.compare(right.similarity, left.similarity);
            if (similarityOrder != 0) {
                return similarityOrder;
            }
            int rightAtomOrder = Integer.compare(left.right, right.right);
            if (rightAtomOrder != 0) {
                return rightAtomOrder;
            }
            return Integer.compare(left.left, right.left);
        }
    };

    /** Port of supplementary IAAPathComparatorChar (AAPath DEFAULT8). */
    static double similarity(Descriptor first, Descriptor second) {
        Descriptor smaller = first;
        Descriptor larger = second;
        if (first.graph.atomTypes.length > second.graph.atomTypes.length) {
            smaller = second;
            larger = first;
        }

        ArrayList<AtomPair> pairs = new ArrayList<AtomPair>(
                smaller.graph.atomTypes.length * larger.graph.atomTypes.length);
        for (int left = 0; left < smaller.graph.atomTypes.length; left++) {
            for (int right = 0; right < larger.graph.atomTypes.length; right++) {
                if (smaller.graph.atomTypes[left] != larger.graph.atomTypes[right]) {
                    continue;
                }
                double atomSimilarity = atomSimilarity(
                        smaller.atomPaths[left], larger.atomPaths[right]);
                if (atomSimilarity > 0.0) {
                    pairs.add(new AtomPair(left, right, atomSimilarity));
                }
            }
        }
        Collections.sort(pairs, ATOM_PAIR_ORDER);

        boolean[] leftAssigned = new boolean[smaller.graph.atomTypes.length];
        boolean[] rightAssigned = new boolean[larger.graph.atomTypes.length];
        double sum = 0.0;
        for (AtomPair pair : pairs) {
            if (!leftAssigned[pair.left] && !rightAssigned[pair.right]) {
                leftAssigned[pair.left] = true;
                rightAssigned[pair.right] = true;
                sum += pair.similarity;
            }
        }
        return sum / (2.0 * Math.max(
                smaller.graph.atomTypes.length, larger.graph.atomTypes.length) - sum);
    }

    private static double atomSimilarity(char[] first, char[] second) {
        if (first.length == 0 || second.length == 0) {
            if (first.length == 0 && second.length == 0) {
                return 1.0;
            }
            int nonemptyLength = Math.max(first.length, second.length);
            return 1.0 / (1.0 + nonemptyLength);
        }

        int left = 0;
        int right = 0;
        int common = 0;
        while (left < first.length && right < second.length) {
            if (first[left] == second[right]) {
                common++;
                left++;
                right++;
            } else if (first[left] < second[right]) {
                left++;
            } else {
                right++;
            }
        }
        return (common + 1.0) /
                (2.0 * Math.max(first.length, second.length) - common + 1.0);
    }

    private static final class Assignment {
        final int clusterIndex;
        final double similarity;

        Assignment(int clusterIndex, double similarity) {
            this.clusterIndex = clusterIndex;
            this.similarity = similarity;
        }
    }

    private static final class RunResult {
        final List<Graph> orderedGraphs;
        final List<Descriptor> descriptors;
        final List<Integer> centroids;
        final Assignment[] assignments;
        final double orderingMs;
        final double descriptorsMs;
        final double selectionMs;
        final double assignmentMs;

        RunResult(List<Graph> orderedGraphs, List<Descriptor> descriptors,
                  List<Integer> centroids, Assignment[] assignments, double orderingMs,
                  double descriptorsMs,
                  double selectionMs, double assignmentMs) {
            this.orderedGraphs = orderedGraphs;
            this.descriptors = descriptors;
            this.centroids = centroids;
            this.assignments = assignments;
            this.orderingMs = orderingMs;
            this.descriptorsMs = descriptorsMs;
            this.selectionMs = selectionMs;
            this.assignmentMs = assignmentMs;
        }

        double workflowMs() {
            return orderingMs + descriptorsMs + selectionMs + assignmentMs;
        }
    }

    private static final class PrioritizedGraph {
        final Graph graph;
        final boolean missing;
        final double priority;

        PrioritizedGraph(Graph graph) {
            this.graph = graph;
            String cleaned = graph.priority == null ? "" : graph.priority.replace("<", "");
            this.missing = cleaned.isEmpty();
            this.priority = missing ? 0.0 : Double.parseDouble(cleaned);
        }
    }

    private static List<Graph> orderByPriority(List<Graph> graphs) {
        ArrayList<PrioritizedGraph> prioritized = new ArrayList<PrioritizedGraph>(graphs.size());
        for (Graph graph : graphs) {
            prioritized.add(new PrioritizedGraph(graph));
        }
        Collections.sort(prioritized, new Comparator<PrioritizedGraph>() {
            @Override
            public int compare(PrioritizedGraph left, PrioritizedGraph right) {
                if (left.missing != right.missing) {
                    return left.missing ? 1 : -1;
                }
                return Double.compare(left.priority, right.priority);
            }
        });
        ArrayList<Graph> ordered = new ArrayList<Graph>(graphs.size());
        for (PrioritizedGraph item : prioritized) {
            ordered.add(item.graph);
        }
        return ordered;
    }

    private static List<Integer> selectCentroids(
            List<Descriptor> descriptors, double threshold, boolean reportProgress) {
        ArrayList<Integer> centroids = new ArrayList<Integer>();
        for (int candidate = 0; candidate < descriptors.size(); candidate++) {
            boolean excluded = false;
            // The published command uses SphereExclusion's default reverse order.
            for (int centroid = centroids.size() - 1; centroid >= 0; centroid--) {
                if (similarity(descriptors.get(centroids.get(centroid)), descriptors.get(candidate))
                        >= threshold) {
                    excluded = true;
                    break;
                }
            }
            if (!excluded) {
                centroids.add(Integer.valueOf(candidate));
            }
            if (reportProgress
                    && ((candidate + 1) % 250 == 0 || candidate + 1 == descriptors.size())) {
                System.out.printf("PROGRESS phase=selection molecules=%d centroids=%d%n",
                        candidate + 1, centroids.size());
            }
        }
        return centroids;
    }

    private static Assignment nearestCentroid(
            int moleculeIndex, List<Descriptor> descriptors, List<Integer> centroids) {
        double bestSimilarity = -1.0;
        int bestCentroid = -1;
        for (int centroid = 0; centroid < centroids.size(); centroid++) {
            double candidateSimilarity = similarity(
                    descriptors.get(centroids.get(centroid)), descriptors.get(moleculeIndex));
            if (candidateSimilarity > bestSimilarity) {
                bestSimilarity = candidateSimilarity;
                bestCentroid = centroid;
            }
        }
        return new Assignment(bestCentroid, bestSimilarity);
    }

    private static List<Graph> readGraphs(Path input) throws IOException {
        ArrayList<Graph> graphs = new ArrayList<Graph>();
        try (BufferedReader reader = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isEmpty() || line.charAt(0) == '#') {
                    continue;
                }
                String[] fields = line.split("\\t", -1);
                if (fields.length != 3 && fields.length != 4) {
                    throw new IOException("Expected three or four tab-separated fields: " + line);
                }
                int inputIndex = Integer.parseInt(fields[0]);
                String[] atomFields = fields[1].split(",");
                int[] atomTypes = new int[atomFields.length];
                for (int atom = 0; atom < atomFields.length; atom++) {
                    atomTypes[atom] = Integer.parseInt(atomFields[atom]);
                }
                int[][] bonds;
                if (fields[2].isEmpty()) {
                    bonds = new int[0][3];
                } else {
                    String[] bondFields = fields[2].split(";");
                    bonds = new int[bondFields.length][3];
                    for (int bond = 0; bond < bondFields.length; bond++) {
                        String[] values = bondFields[bond].split(",");
                        if (values.length != 3) {
                            throw new IOException("Malformed bond: " + bondFields[bond]);
                        }
                        for (int item = 0; item < 3; item++) {
                            bonds[bond][item] = Integer.parseInt(values[item]);
                        }
                    }
                }
                graphs.add(new Graph(inputIndex, fields.length == 4 ? fields[3] : null, atomTypes, bonds));
            }
        }
        return graphs;
    }

    private static RunResult clusterOnce(
            List<Graph> graphs, double threshold, int maxBonds, int threads,
            boolean orderByPriority, boolean reportProgress) throws Exception {
        long orderingStart = System.nanoTime();
        List<Graph> orderedGraphs = orderByPriority ? orderByPriority(graphs) : graphs;
        long descriptorStart = System.nanoTime();
        ArrayList<Descriptor> descriptors = new ArrayList<Descriptor>(orderedGraphs.size());
        for (int graph = 0; graph < orderedGraphs.size(); graph++) {
            descriptors.add(new Descriptor(orderedGraphs.get(graph), maxBonds));
            if (reportProgress
                    && ((graph + 1) % 250 == 0 || graph + 1 == orderedGraphs.size())) {
                System.out.printf("PROGRESS phase=descriptors molecules=%d%n", graph + 1);
            }
        }
        long selectionStart = System.nanoTime();
        List<Integer> centroids = selectCentroids(descriptors, threshold, reportProgress);
        long assignmentStart = System.nanoTime();

        final Assignment[] assignments = new Assignment[orderedGraphs.size()];
        for (int cluster = 0; cluster < centroids.size(); cluster++) {
            assignments[centroids.get(cluster)] = new Assignment(cluster, 1.0);
        }
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        ArrayList<Future<Void>> futures = new ArrayList<Future<Void>>();
        for (int molecule = 0; molecule < orderedGraphs.size(); molecule++) {
            if (assignments[molecule] != null) {
                continue;
            }
            final int moleculeIndex = molecule;
            futures.add(executor.submit(new Callable<Void>() {
                @Override
                public Void call() {
                    assignments[moleculeIndex] = nearestCentroid(
                            moleculeIndex, descriptors, centroids);
                    return null;
                }
            }));
        }
        int completedAssignments = 0;
        for (Future<Void> future : futures) {
            future.get();
            completedAssignments++;
            if (reportProgress && (completedAssignments % 250 == 0
                    || completedAssignments == futures.size())) {
                System.out.printf("PROGRESS phase=nearest_assignment molecules=%d%n",
                        completedAssignments);
            }
        }
        executor.shutdown();
        long finish = System.nanoTime();
        return new RunResult(
                orderedGraphs, descriptors, centroids, assignments,
                (descriptorStart - orderingStart) / 1.0e6,
                (selectionStart - descriptorStart) / 1.0e6,
                (assignmentStart - selectionStart) / 1.0e6,
                (finish - assignmentStart) / 1.0e6);
    }

    private static void writeAssignments(
            List<Graph> graphs, RunResult result, Path outputPath) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8)) {
            writer.write("input_index\tcluster_index\tcentroid_input_index\tsimilarity\tis_centroid\n");
            for (int molecule = 0; molecule < graphs.size(); molecule++) {
                Assignment assignment = result.assignments[molecule];
                int centroidInput = graphs.get(
                        result.centroids.get(assignment.clusterIndex)).inputIndex;
                writer.write(Integer.toString(graphs.get(molecule).inputIndex));
                writer.write('\t');
                writer.write(Integer.toString(assignment.clusterIndex));
                writer.write('\t');
                writer.write(Integer.toString(centroidInput));
                writer.write('\t');
                writer.write(Double.toString(assignment.similarity));
                writer.write('\t');
                writer.write(result.assignments[molecule].similarity == 1.0
                        && result.centroids.get(assignment.clusterIndex) == molecule ? "1\n" : "0\n");
            }
        }
    }

    private static void runCluster(String[] args) throws Exception {
        if (args.length != 6) {
            throw new IllegalArgumentException(
                    "cluster <graphs.tsv> <threshold> <max-bonds> <threads> <output.tsv>");
        }
        Path graphPath = Paths.get(args[1]);
        double threshold = Double.parseDouble(args[2]);
        int maxBonds = Integer.parseInt(args[3]);
        int threads = Integer.parseInt(args[4]);
        Path outputPath = Paths.get(args[5]);

        long loadStart = System.nanoTime();
        List<Graph> graphs = readGraphs(graphPath);
        long computeStart = System.nanoTime();
        RunResult result = clusterOnce(graphs, threshold, maxBonds, threads, false, true);
        long writeStart = System.nanoTime();
        writeAssignments(result.orderedGraphs, result, outputPath);
        long finish = System.nanoTime();
        System.out.printf(
                "METRIC molecules=%d centroids=%d load_ms=%.3f descriptors_ms=%.3f "
                        + "selection_ms=%.3f assignment_ms=%.3f compute_ms=%.3f "
                        + "write_ms=%.3f total_ms=%.3f%n",
                graphs.size(), result.centroids.size(),
                (computeStart - loadStart) / 1.0e6,
                result.descriptorsMs, result.selectionMs, result.assignmentMs,
                result.workflowMs(), (finish - writeStart) / 1.0e6,
                (finish - loadStart) / 1.0e6);
    }

    private static void runBenchmark(String[] args) throws Exception {
        if (args.length != 8) {
            throw new IllegalArgumentException(
                    "benchmark <graphs.tsv> <threshold> <max-bonds> <threads> "
                            + "<warmups> <runs> <output.tsv>");
        }
        Path graphPath = Paths.get(args[1]);
        double threshold = Double.parseDouble(args[2]);
        int maxBonds = Integer.parseInt(args[3]);
        int threads = Integer.parseInt(args[4]);
        int warmups = Integer.parseInt(args[5]);
        int runs = Integer.parseInt(args[6]);
        Path outputPath = Paths.get(args[7]);

        boolean sortByPriority = "benchmark-workflow".equals(args[0]);
        List<Graph> graphs = readGraphs(graphPath);
        for (int warmup = 0; warmup < warmups; warmup++) {
            RunResult result = clusterOnce(graphs, threshold, maxBonds, threads, sortByPriority, false);
            System.out.printf("WARMUP run=%d workflow_ms=%.3f centroids=%d%n",
                    warmup + 1, result.workflowMs(), result.centroids.size());
        }

        RunResult last = null;
        for (int run = 0; run < runs; run++) {
            last = clusterOnce(graphs, threshold, maxBonds, threads, sortByPriority, false);
            System.out.printf(
                    "BENCHMARK run=%d molecules=%d centroids=%d ordering_ms=%.3f "
                            + "descriptors_ms=%.3f selection_ms=%.3f assignment_ms=%.3f "
                            + "workflow_ms=%.3f%n",
                    run + 1, graphs.size(), last.centroids.size(), last.orderingMs,
                    last.descriptorsMs, last.selectionMs, last.assignmentMs, last.workflowMs());
        }
        if (last == null) {
            throw new IllegalArgumentException("runs must be at least one");
        }
        writeAssignments(last.orderedGraphs, last, outputPath);
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            throw new IllegalArgumentException("Usage: GCheminfoAAPDISE cluster|benchmark ...");
        }
        if ("cluster".equals(args[0])) {
            runCluster(args);
        } else if ("benchmark".equals(args[0]) || "benchmark-workflow".equals(args[0])) {
            runBenchmark(args);
        } else {
            throw new IllegalArgumentException("Unknown command: " + args[0]);
        }
    }

    private GCheminfoAAPDISE() {}
}
