#!/usr/bin/perl -w 

if (scalar(@ARGV) ne 3)
{
  print STDERR "perl script FIND max-length input   ---OR---\n".
               "perl script GENERATE id input\n";
  exit(1);
}



if ($ARGV[0] eq "FIND")
{
  $line_num = 0;
  $max_len = $ARGV[1];
  open(INPUT, "<", $ARGV[2]) or die "Error: can not read $ARGV[0]!\n";
  open(OUTPUT, ">", $ARGV[2].".max$max_len.id") or die "Error: can not write $ARGV[0].max$max_len!\n";

  $max_len_num = 0;
  while(my $line = <INPUT>)
  {
    ++$line_num;
    $line =~ s/[\r\n]//g;
    my @words = split /\s/, $line;
    if (scalar(@words) <= $max_len) 
    {
      ++$max_len_num;
      print OUTPUT $line_num."\n";
    }
  }
  print STDERR "MAX_LEN=$max_len MAX_LEN_NUM=$max_len_num\n";
  close(INPUT);
  close(OUTPUT);
}
elsif ($ARGV[0] eq "GENERATE")
{
  my %id_hash;
  open (ID, "<", $ARGV[1]) or die "Error: can not read $ARGV[1]!\n";
  while (my $line = <ID>)
  {
    $line =~ s/[\r\n]//g;
    $line =~ s/^\s+//g;
    $line =~ s/\s+$//g;
    $id_hash{$line}++;
  }  
  my $id_size = keys %id_hash;
  print STDERR "id_size=$id_size\n"; 
  close(ID);
 
  open (INPUT, "<", $ARGV[2]) or die "Error: can not read $ARGV[2]!\n";
  open (OUTPUT, ">", $ARGV[2].".out") or die "Error: can not write $ARGV[2].out!\n";
  my $line_num = 0;
  while (my $line = <INPUT>)
  {
    ++$line_num;
    $line =~ s/[\r\n]//g;
    $line =~ s/^\s+//g;
    $line =~ s/\s+$//g;
    if (exists $id_hash{$line_num})
    {
      print OUTPUT $line."\n";
    }
  }

  close(INPUT);
  close(OUTPUT);

}
else
{
  print STDERR "The first parameter must be FIND|GENERATE\n";
  exit(1);
}

